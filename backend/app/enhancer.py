import os
import time
import uuid
import subprocess
import cv2
import numpy as np
from app.cpu_engine import CPUEngine
from app.cuda_engine import CUDAEngine

class EnhancementManager:
    """Unified enhancement manager coordinating CPU and CUDA engines for images and videos."""

    def __init__(self, cuda_engine: CUDAEngine = None):
        self.cuda_engine = cuda_engine if cuda_engine else CUDAEngine()
        self.video_jobs = {}

    def get_backend_status(self) -> dict:
        return self.cuda_engine.get_hardware_status()

    def enhance_image(
        self,
        img: np.ndarray,
        gamma: float = 2.0,
        brightness: float = 10.0,
        contrast: float = 1.25,
        denoising_strength: float = 0.5,
        sharpening_strength: float = 0.5,
        use_clahe: bool = False,
        auto_enhance: bool = False,
        backend_preference: str = "auto"
    ) -> tuple[np.ndarray, dict]:
        """
        Enhances a single image based on parameters and requested backend.
        backend_preference: 'auto', 'cuda', 'cpu'
        """
        # Determine active backend
        use_cuda = False
        if backend_preference in ["auto", "cuda"] and self.cuda_engine.gpu_available:
            use_cuda = True

        t_total_start = time.perf_counter()

        if auto_enhance:
            # Auto calculate parameters if requested
            img_auto, calc_gamma, calc_contrast = CPUEngine.apply_auto_brightness(img)
            gamma = calc_gamma
            contrast = calc_contrast

        if use_cuda:
            enhanced_img, metrics = self.cuda_engine.apply_auto_enhance(
                img, gamma=gamma, brightness=brightness, contrast=contrast,
                denoising_strength=denoising_strength, sharpening_strength=sharpening_strength
            )
            # If CLAHE requested, apply CLAHE on CPU output
            if use_clahe:
                enhanced_img = CPUEngine.apply_histogram_equalization(enhanced_img, use_clahe=True)
            
            metrics["active_backend"] = "CUDA GPU"
            metrics["gamma_used"] = round(gamma, 2)
            metrics["contrast_used"] = round(contrast, 2)
            return enhanced_img, metrics
        else:
            t0 = time.perf_counter()
            enhanced_img = CPUEngine.apply_auto_enhance(
                img, gamma=gamma, brightness=brightness, contrast=contrast,
                denoising_strength=denoising_strength, sharpening_strength=sharpening_strength,
                use_clahe=use_clahe
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            
            metrics = {
                "kernel_time_ms": round(duration_ms, 4),
                "htod_ms": 0.0,
                "dtoh_ms": 0.0,
                "total_time_ms": round(duration_ms, 4),
                "backend": "CPU Engine",
                "active_backend": "CPU",
                "gamma_used": round(gamma, 2),
                "contrast_used": round(contrast, 2)
            }
            return enhanced_img, metrics

    def start_video_job(
        self,
        job_id: str,
        input_path: str,
        output_path: str,
        params: dict
    ):
        """Starts asynchronous video frame-by-frame enhancement background job."""
        self.video_jobs[job_id] = {
            "status": "processing",
            "progress_percentage": 0.0,
            "processed_frames": 0,
            "total_frames": 0,
            "fps": 0.0,
            "processing_fps": 0.0,
            "duration_sec": 0.0,
            "backend": "CUDA GPU" if (params.get("backend_preference", "auto") != "cpu" and self.cuda_engine.gpu_available) else "CPU",
            "output_path": output_path,
            "error": None
        }

        import threading
        thread = threading.Thread(
            target=self._process_video_thread,
            args=(job_id, input_path, output_path, params),
            daemon=True
        )
        thread.start()

    def _process_video_thread(self, job_id: str, input_path: str, output_path: str, params: dict):
        job = self.video_jobs[job_id]
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError("Failed to open input video file.")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            job["total_frames"] = total_frames
            job["fps"] = fps
            job["duration_sec"] = total_frames / fps if fps > 0 else 0.0

            # Select video encoder codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = output_path.replace(".mp4", "_temp.mp4")
            writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            gamma = float(params.get("gamma", 2.0))
            brightness = float(params.get("brightness", 10.0))
            contrast = float(params.get("contrast", 1.25))
            denoising = float(params.get("denoising_strength", 0.5))
            sharpening = float(params.get("sharpening_strength", 0.5))
            backend_pref = params.get("backend_preference", "auto")

            use_cuda = (backend_pref in ["auto", "cuda"]) and self.cuda_engine.gpu_available

            processed = 0
            t_start = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if use_cuda:
                    enhanced_frame, _ = self.cuda_engine.apply_auto_enhance(
                        frame, gamma=gamma, brightness=brightness, contrast=contrast,
                        denoising_strength=denoising, sharpening_strength=sharpening
                    )
                else:
                    enhanced_frame = CPUEngine.apply_auto_enhance(
                        frame, gamma=gamma, brightness=brightness, contrast=contrast,
                        denoising_strength=denoising, sharpening_strength=sharpening
                    )

                writer.write(enhanced_frame)
                processed += 1

                elapsed = time.perf_counter() - t_start
                proc_fps = processed / elapsed if elapsed > 0 else 0.0
                progress = (processed / total_frames * 100.0) if total_frames > 0 else 0.0

                job["processed_frames"] = processed
                job["progress_percentage"] = min(round(progress, 2), 100.0)
                job["processing_fps"] = round(proc_fps, 2)

            cap.release()
            writer.release()

            # Attempt audio muxing with FFmpeg if installed
            self._try_mux_audio(input_path, temp_output, output_path)

            job["status"] = "completed"
            job["progress_percentage"] = 100.0
            print(f"[VideoJob {job_id}] Video processing complete! {processed} frames processed.")

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            print(f"[VideoJob {job_id}] Error: {e}")

    def _try_mux_audio(self, src_video: str, enhanced_video: str, final_video: str):
        """Attempts to mux original audio stream into enhanced video using FFmpeg if available."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", enhanced_video,
                "-i", src_video,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                final_video
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode == 0 and os.path.exists(final_video) and os.path.getsize(final_video) > 0:
                os.remove(enhanced_video)
                return
        except Exception:
            pass

        # Fallback: rename temp enhanced video to final destination
        if os.path.exists(enhanced_video):
            if os.path.exists(final_video):
                os.remove(final_video)
            os.rename(enhanced_video, final_video)

    def get_video_job_status(self, job_id: str) -> dict:
        return self.video_jobs.get(job_id, {"status": "not_found"})
