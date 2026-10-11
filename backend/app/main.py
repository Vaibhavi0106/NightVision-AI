import os
import time
import uuid
import shutil
import zipfile
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.cuda_engine import CUDAEngine
from app.enhancer import EnhancementManager
from app.benchmarker import PerformanceBenchmarker

app = FastAPI(
    title="NightVision AI Backend API",
    description="GPU-Accelerated Low-Light Photo & Video Enhancement Platform API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist")

# Initialize Core Engines
cuda_engine = CUDAEngine()
enhancer = EnhancementManager(cuda_engine=cuda_engine)
benchmarker = PerformanceBenchmarker(cuda_engine=cuda_engine)

# In-memory session statistics
session_stats = {
    "total_files_processed": 0,
    "total_processing_time_ms": 0.0,
    "recent_jobs": []
}

def record_session_job(job_type: str, file_name: str, duration_ms: float, backend: str, dimensions: str):
    session_stats["total_files_processed"] += 1
    session_stats["total_processing_time_ms"] += duration_ms
    
    job_record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%H:%M:%S"),
        "type": job_type,
        "name": file_name,
        "duration_ms": round(duration_ms, 2),
        "backend": backend,
        "dimensions": dimensions
    }
    session_stats["recent_jobs"].insert(0, job_record)
    if len(session_stats["recent_jobs"]) > 10:
        session_stats["recent_jobs"].pop()

@app.get("/api/health")
def health_check():
    hw = cuda_engine.get_hardware_status()
    avg_time = (
        session_stats["total_processing_time_ms"] / session_stats["total_files_processed"]
        if session_stats["total_files_processed"] > 0 else 0.0
    )
    return {
        "status": "online",
        "hardware": hw,
        "stats": {
            "total_files_processed": session_stats["total_files_processed"],
            "avg_processing_time_ms": round(avg_time, 2),
            "recent_jobs": session_stats["recent_jobs"]
        }
    }

@app.get("/api/files/{file_id}")
def serve_file(file_id: str):
    safe_name = os.path.basename(file_id)
    target = os.path.join(TEMP_DIR, safe_name)
    if not os.path.exists(target):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_images", safe_name)
        if os.path.exists(sample_path):
            return FileResponse(sample_path)
        raise HTTPException(status_code=404, detail="Requested file not found.")
    return FileResponse(target)

@app.post("/api/image/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image file format.")

    file_id = f"{uuid.uuid4().hex}{ext}"
    target_path = os.path.join(TEMP_DIR, file_id)

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(target_path)
    if img is None:
        os.remove(target_path)
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    h, w, c = img.shape
    return {
        "file_id": file_id,
        "filename": file.filename,
        "width": w,
        "height": h,
        "channels": c,
        "preview_url": f"/api/files/{file_id}"
    }

@app.get("/api/image/sample")
def get_sample_image():
    sample_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_images", "sample_dark.jpg")
    if not os.path.exists(sample_src):
        from create_sample_images import generate_sample_images
        generate_sample_images()

    file_id = f"sample_{uuid.uuid4().hex[:6]}.jpg"
    target_path = os.path.join(TEMP_DIR, file_id)
    shutil.copy(sample_src, target_path)

    img = cv2.imread(target_path)
    h, w, c = img.shape
    return {
        "file_id": file_id,
        "filename": "sample_dark.jpg",
        "width": w,
        "height": h,
        "channels": c,
        "preview_url": f"/api/files/{file_id}"
    }

@app.post("/api/image/enhance")
def enhance_image(
    file_id: str = Form(...),
    gamma: float = Form(2.0),
    brightness: float = Form(10.0),
    contrast: float = Form(1.25),
    denoising_strength: float = Form(0.5),
    sharpening_strength: float = Form(0.5),
    use_clahe: bool = Form(False),
    auto_enhance: bool = Form(False),
    backend_preference: str = Form("auto")
):
    safe_name = os.path.basename(file_id)
    input_path = os.path.join(TEMP_DIR, safe_name)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Uploaded source image not found.")

    img = cv2.imread(input_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    enhanced_img, metrics = enhancer.enhance_image(
        img,
        gamma=gamma,
        brightness=brightness,
        contrast=contrast,
        denoising_strength=denoising_strength,
        sharpening_strength=sharpening_strength,
        use_clahe=use_clahe,
        auto_enhance=auto_enhance,
        backend_preference=backend_preference
    )

    out_file_id = f"enhanced_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(TEMP_DIR, out_file_id)
    cv2.imwrite(output_path, enhanced_img)

    h, w, c = enhanced_img.shape
    record_session_job("Image", file_id, metrics["total_time_ms"], metrics["active_backend"], f"{w}x{h}")

    return {
        "file_id": out_file_id,
        "original_url": f"/api/files/{file_id}",
        "enhanced_url": f"/api/files/{out_file_id}",
        "width": w,
        "height": h,
        "metrics": metrics
    }

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    file_id = f"{uuid.uuid4().hex}{ext}"
    target_path = os.path.join(TEMP_DIR, file_id)

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        os.remove(target_path)
        raise HTTPException(status_code=400, detail="Could not read video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0

    return {
        "file_id": file_id,
        "filename": file.filename,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "preview_url": f"/api/files/{file_id}"
    }

@app.post("/api/video/enhance")
def enhance_video(
    file_id: str = Form(...),
    gamma: float = Form(2.0),
    brightness: float = Form(10.0),
    contrast: float = Form(1.25),
    denoising_strength: float = Form(0.5),
    sharpening_strength: float = Form(0.5),
    backend_preference: str = Form("auto")
):
    safe_name = os.path.basename(file_id)
    input_path = os.path.join(TEMP_DIR, safe_name)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Input video file not found.")

    job_id = f"vjob_{uuid.uuid4().hex[:8]}"
    out_file_id = f"enhanced_{job_id}.mp4"
    output_path = os.path.join(TEMP_DIR, out_file_id)

    params = {
        "gamma": gamma,
        "brightness": brightness,
        "contrast": contrast,
        "denoising_strength": denoising_strength,
        "sharpening_strength": sharpening_strength,
        "backend_preference": backend_preference
    }

    enhancer.start_video_job(job_id, input_path, output_path, params)

    return {
        "job_id": job_id,
        "output_file_id": out_file_id,
        "status": "processing"
    }

@app.get("/api/video/status/{job_id}")
def get_video_status(job_id: str):
    status = enhancer.get_video_job_status(job_id)
    if status["status"] == "completed":
        out_path = status["output_path"]
        out_name = os.path.basename(out_path)
        status["result_url"] = f"/api/files/{out_name}"
    return status

@app.post("/api/batch/process")
async def batch_process_images(
    files: list[UploadFile] = File(...),
    gamma: float = Form(2.0),
    brightness: float = Form(10.0),
    contrast: float = Form(1.25),
    denoising_strength: float = Form(0.5),
    sharpening_strength: float = Form(0.5),
    backend_preference: str = Form("auto")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch processing.")

    results = []
    zip_filename = f"nightvision_batch_{uuid.uuid4().hex[:6]}.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zip_out:
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Unsupported image format"
                })
                continue

            content = await file.read()
            np_arr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Could not decode image"
                })
                continue

            enhanced_img, metrics = enhancer.enhance_image(
                img,
                gamma=gamma,
                brightness=brightness,
                contrast=contrast,
                denoising_strength=denoising_strength,
                sharpening_strength=sharpening_strength,
                backend_preference=backend_preference
            )

            out_id = f"batch_{uuid.uuid4().hex[:6]}.png"
            out_file_path = os.path.join(TEMP_DIR, out_id)
            cv2.imwrite(out_file_path, enhanced_img)

            zip_out.write(out_file_path, arcname=f"enhanced_{file.filename}")

            h, w, c = enhanced_img.shape
            record_session_job("Batch Image", file.filename, metrics["total_time_ms"], metrics["active_backend"], f"{w}x{h}")

            results.append({
                "filename": file.filename,
                "status": "success",
                "output_url": f"/api/files/{out_id}",
                "metrics": metrics,
                "dimensions": f"{w}x{h}"
            })

    return {
        "results": results,
        "zip_url": f"/api/files/{zip_filename}",
        "total_files": len(files)
    }

@app.post("/api/benchmark/run")
def run_benchmark(file: UploadFile = File(None)):
    custom_img = None
    if file is not None:
        contents = file.file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        custom_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    bench_data = benchmarker.run_benchmark_suite(custom_img=custom_img)
    return bench_data

# Mount Frontend Dist for Single-Server Convenience
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
