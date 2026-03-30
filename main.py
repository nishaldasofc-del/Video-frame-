"""
Video Frame Extractor — Full Backend
Stack: FastAPI · Celery · Supabase (Postgres + Storage) · OpenCV · PySceneDetect
No Redis, no S3, no card required — 100% free tier
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

import cv2
from celery import Celery
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from supabase import create_client, Client


# ─────────────────────────────────────────────
# 1. SETTINGS
# ─────────────────────────────────────────────

class Settings(BaseSettings):
    # Supabase — get these from your project dashboard → Settings → API
    supabase_url: str = "https://fpvjkstwajychrnmidty.supabase.co"
    supabase_service_key: str = ""          # Settings → API → service_role key

    # Celery broker — uses Supabase Postgres via SQLAlchemy
    # Format: postgresql+psycopg2://postgres:<password>@db.<ref>.supabase.co:5432/postgres
    celery_broker_url: str = ""
    celery_result_backend: str = ""         # same URL as broker

    # Storage
    storage_bucket: str = "video-frames"
    cleanup_ttl_seconds: int = 3600         # 1 hour

    # Extraction defaults
    default_image_format: str = "jpg"
    default_scene_threshold: float = 27.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# ─────────────────────────────────────────────
# 2. SUPABASE CLIENT
# ─────────────────────────────────────────────

def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


# ─────────────────────────────────────────────
# 3. PROGRESS HELPERS (Supabase Postgres)
# ─────────────────────────────────────────────

def set_progress(task_id: str, percent: int, message: str = "") -> None:
    sb = get_supabase()
    sb.table("vfe_progress").upsert({
        "task_id": task_id,
        "percent": percent,
        "message": message,
    }).execute()


def get_progress(task_id: str) -> dict:
    sb = get_supabase()
    res = sb.table("vfe_progress").select("percent,message").eq("task_id", task_id).single().execute()
    if res.data:
        return {"percent": res.data["percent"], "message": res.data["message"]}
    return {"percent": 0, "message": ""}


def set_task_meta(task_id: str, data: dict) -> None:
    sb = get_supabase()
    sb.table("vfe_tasks").upsert({"id": task_id, **data}).execute()


def get_task_meta(task_id: str) -> dict | None:
    sb = get_supabase()
    res = sb.table("vfe_tasks").select("*").eq("id", task_id).single().execute()
    return res.data if res.data else None


# ─────────────────────────────────────────────
# 4. SUPABASE STORAGE HELPERS
# ─────────────────────────────────────────────

def upload_to_storage(path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    sb = get_supabase()
    sb.storage.from_(settings.storage_bucket).upload(
        path=path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return path


def upload_file_to_storage(path: str, filepath: str, content_type: str = "application/octet-stream") -> str:
    with open(filepath, "rb") as f:
        return upload_to_storage(path, f.read(), content_type)


def get_signed_url(path: str, expires_in: int = 3600) -> str:
    sb = get_supabase()
    res = sb.storage.from_(settings.storage_bucket).create_signed_url(path, expires_in)
    return res["signedURL"]


def delete_storage_folder(prefix: str) -> None:
    sb = get_supabase()
    files = sb.storage.from_(settings.storage_bucket).list(prefix)
    if files:
        paths = [f"{prefix}/{f['name']}" for f in files]
        sb.storage.from_(settings.storage_bucket).remove(paths)


# ─────────────────────────────────────────────
# 5. CELERY (Supabase Postgres as broker)
# ─────────────────────────────────────────────

celery_app = Celery(
    "vfe",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_soft_time_limit=1800,
    task_time_limit=2100,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=7200,
)


# ─────────────────────────────────────────────
# 6. FRAME EXTRACTOR + SCENE DETECTION
# ─────────────────────────────────────────────

def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": round(total_frames / fps if fps > 0 else 0, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def detect_scene_boundaries(video_path: str, threshold: float) -> set[int]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    return {scene[0].get_frames() for scene in scene_manager.get_scene_list()}


def extract_and_upload_frames(
    task_id: str,
    video_path: str,
    image_format: str,
    frame_step: int,
    resize: tuple[int, int] | None,
    scene_detect: bool,
    scene_threshold: float,
) -> dict:
    set_progress(task_id, 0, "Opening video...")
    metadata = get_video_metadata(video_path)
    total_frames = metadata["total_frames"]

    if scene_detect:
        set_progress(task_id, 2, "Detecting scene boundaries...")
        scene_frames = detect_scene_boundaries(video_path, scene_threshold)
    else:
        scene_frames = set()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths: list[str] = []
    count = 0
    saved = 0
    ext = image_format.lower()
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == "jpg" else []

    set_progress(task_id, 5, "Extracting frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_save = (
            (scene_detect and count in scene_frames)
            or (not scene_detect and count % frame_step == 0)
        )

        if should_save:
            if resize:
                frame = cv2.resize(frame, resize)
            success, buf = cv2.imencode(f".{ext}", frame, encode_param)
            if success:
                storage_path = f"{task_id}/frames/frame_{saved:06d}.{ext}"
                upload_to_storage(storage_path, buf.tobytes(), f"image/{ext}")
                frame_paths.append(storage_path)
                saved += 1

        count += 1
        if count % 30 == 0:
            pct = min(5 + int((count / total_frames) * 80), 85)
            set_progress(task_id, pct, f"Extracting... {saved} frames saved")

    cap.release()

    # Build ZIP from uploaded frames
    set_progress(task_id, 87, f"Building ZIP ({saved} frames)...")
    sb = get_supabase()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, path in enumerate(frame_paths):
            res = sb.storage.from_(settings.storage_bucket).download(path)
            zf.writestr(path.split("/")[-1], res)
            if i % 20 == 0:
                pct = 87 + int((i / max(len(frame_paths), 1)) * 10)
                set_progress(task_id, min(pct, 97), "Compressing...")

    zip_path = f"{task_id}/frames.zip"
    upload_to_storage(zip_path, zip_buffer.getvalue(), "application/zip")
    set_progress(task_id, 100, "Done")

    return {
        "frames_extracted": saved,
        "zip_storage_path": zip_path,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────
# 7. CELERY TASKS
# ─────────────────────────────────────────────

@celery_app.task(bind=True, name="vfe.process_video", max_retries=2)
def process_video_task(
    self,
    task_id: str,
    video_storage_path: str,
    image_format: str,
    frame_step: int,
    resize: list[int] | None,
    scene_detect: bool,
    scene_threshold: float,
):
    tmp_dir = tempfile.mkdtemp(prefix="vfe_")
    try:
        # Download video from Supabase Storage to tmp
        set_progress(task_id, 1, "Downloading video...")
        sb = get_supabase()
        video_bytes = sb.storage.from_(settings.storage_bucket).download(video_storage_path)
        local_video = os.path.join(tmp_dir, "source_video")
        with open(local_video, "wb") as f:
            f.write(video_bytes)

        result = extract_and_upload_frames(
            task_id=task_id,
            video_path=local_video,
            image_format=image_format,
            frame_step=frame_step,
            resize=tuple(resize) if resize else None,
            scene_detect=scene_detect,
            scene_threshold=scene_threshold,
        )

        set_task_meta(task_id, {
            "status": "completed",
            "frames_extracted": result["frames_extracted"],
            "zip_storage_path": result["zip_storage_path"],
            "metadata": result["metadata"],
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as exc:
        set_progress(task_id, -1, f"Error: {exc}")
        set_task_meta(task_id, {"status": "failed", "error": str(exc)})
        raise self.retry(exc=exc, countdown=10)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@celery_app.task(name="vfe.cleanup_old_tasks")
def cleanup_old_tasks_task():
    sb = get_supabase()
    res = sb.rpc("cleanup_old_tasks", {"ttl_seconds": settings.cleanup_ttl_seconds}).execute()
    return {"deleted": res.data}


# ─────────────────────────────────────────────
# 8. PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    percent: int
    message: str
    metadata: dict | None = None
    frames_extracted: int | None = None
    error: str | None = None


class DownloadResponse(BaseModel):
    task_id: str
    download_url: str
    expires_in_seconds: int
    frames_extracted: int


# ─────────────────────────────────────────────
# 9. FASTAPI APP
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify Supabase connection on startup
    sb = get_supabase()
    sb.table("vfe_tasks").select("id").limit(1).execute()
    yield


app = FastAPI(
    title="Video Frame Extractor API",
    description="Upload video → extract frames → download ZIP. Free stack: Supabase + Railway.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /upload ──────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    image_format: str = Form("jpg"),
    frame_step: int = Form(1, ge=1),
    resize_width: Optional[int] = Form(None, ge=1),
    resize_height: Optional[int] = Form(None, ge=1),
    scene_detect: bool = Form(False),
    scene_threshold: float = Form(27.0, ge=1.0, le=100.0),
):
    task_id = uuid.uuid4().hex[:12]
    resize = [resize_width, resize_height] if resize_width and resize_height else None

    # Upload video directly to Supabase Storage
    video_bytes = await file.read()
    ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "mp4"
    video_storage_path = f"{task_id}/source/video.{ext}"
    upload_to_storage(video_storage_path, video_bytes, file.content_type or "video/mp4")

    # Create task record
    set_task_meta(task_id, {
        "filename": file.filename,
        "status": "queued",
        "image_format": image_format,
        "frame_step": frame_step,
        "resize": resize,
        "scene_detect": scene_detect,
        "scene_threshold": scene_threshold,
        "video_storage_path": video_storage_path,
    })
    set_progress(task_id, 0, "Queued")

    # Dispatch Celery task
    process_video_task.apply_async(
        kwargs=dict(
            task_id=task_id,
            video_storage_path=video_storage_path,
            image_format=image_format,
            frame_step=frame_step,
            resize=resize,
            scene_detect=scene_detect,
            scene_threshold=scene_threshold,
        ),
        task_id=task_id,
    )

    return UploadResponse(task_id=task_id, status="queued", message="Video uploaded. Processing started.")


# ── GET /status/{task_id} ─────────────────────

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    meta = get_task_meta(task_id)
    if not meta:
        raise HTTPException(404, "Task not found")
    prog = get_progress(task_id)
    return StatusResponse(
        task_id=task_id,
        status=meta.get("status", "unknown"),
        percent=prog["percent"],
        message=prog["message"],
        metadata=meta.get("metadata"),
        frames_extracted=meta.get("frames_extracted"),
        error=meta.get("error"),
    )


# ── GET /info/{task_id} ───────────────────────

@app.get("/info/{task_id}")
async def get_info(task_id: str):
    meta = get_task_meta(task_id)
    if not meta:
        raise HTTPException(404, "Task not found")
    if not meta.get("metadata"):
        raise HTTPException(202, "Metadata not ready yet")
    return {"task_id": task_id, **meta["metadata"]}


# ── GET /progress/{task_id} — SSE ────────────

@app.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    if not get_task_meta(task_id):
        raise HTTPException(404, "Task not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_pct = -1
        while True:
            prog = get_progress(task_id)
            pct = prog["percent"]
            msg = prog["message"]

            if pct != last_pct:
                payload = json.dumps({"task_id": task_id, "percent": pct, "message": msg})
                yield f"data: {payload}\n\n"
                last_pct = pct

            if pct >= 100 or pct < 0:
                meta = get_task_meta(task_id) or {}
                final = json.dumps({
                    "task_id": task_id,
                    "percent": pct,
                    "status": meta.get("status", "unknown"),
                    "frames_extracted": meta.get("frames_extracted"),
                })
                yield f"event: complete\ndata: {final}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── GET /download/{task_id} ───────────────────

@app.get("/download/{task_id}", response_model=DownloadResponse)
async def download_frames(task_id: str, expires_in: int = 3600):
    meta = get_task_meta(task_id)
    if not meta:
        raise HTTPException(404, "Task not found")
    if meta.get("status") != "completed":
        raise HTTPException(202, f"Not ready — status: {meta.get('status')}")
    zip_path = meta.get("zip_storage_path")
    if not zip_path:
        raise HTTPException(500, "ZIP path missing")
    url = get_signed_url(zip_path, expires_in)
    return DownloadResponse(
        task_id=task_id,
        download_url=url,
        expires_in_seconds=expires_in,
        frames_extracted=meta.get("frames_extracted", 0),
    )


# ── DELETE /task/{task_id} ────────────────────

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    meta = get_task_meta(task_id)
    if not meta:
        raise HTTPException(404, "Task not found")
    celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")
    delete_storage_folder(task_id)
    sb = get_supabase()
    sb.table("vfe_tasks").delete().eq("id", task_id).execute()
    return {"task_id": task_id, "status": "deleted"}


# ── POST /cleanup ─────────────────────────────

@app.post("/cleanup")
async def trigger_cleanup():
    result = cleanup_old_tasks_task.delay()
    return {"cleanup_task_id": result.id, "status": "triggered"}


# ── GET /health ───────────────────────────────

@app.get("/health")
async def health():
    try:
        get_supabase().table("vfe_tasks").select("id").limit(1).execute()
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "supabase": "up" if db_ok else "down",
        "bucket": settings.storage_bucket,
    }
