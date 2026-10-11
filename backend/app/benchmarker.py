import time
import numpy as np
from app.cpu_engine import CPUEngine
from app.cuda_engine import CUDAEngine

class PerformanceBenchmarker:
    """Performance Lab benchmarking suite comparing CPU vs CUDA GPU implementations across image resolutions."""

    def __init__(self, cuda_engine: CUDAEngine = None):
        self.cuda_engine = cuda_engine if cuda_engine else CUDAEngine()

    def run_benchmark_suite(self, custom_img: np.ndarray = None, iterations: int = 3) -> dict:
        """
        Runs comprehensive benchmark suite across standard resolutions (720p, 1080p, 1440p, 4K)
        or on a custom uploaded image.
        """
        resolutions = {
            "720p": (720, 1280),
            "1080p": (1080, 1920),
            "1440p": (1440, 2560),
            "4K": (2160, 3840)
        }

        results = []

        if custom_img is not None:
            height, width = custom_img.shape[:2]
            label = f"Custom ({width}x{height})"
            bench_res = self._benchmark_single_resolution(custom_img, label, iterations)
            results.append(bench_res)
        else:
            for res_name, (h, w) in resolutions.items():
                # Generate synthetic low-light image for benchmarking
                synth_img = np.random.randint(5, 80, (h, w, 3), dtype=np.uint8)
                bench_res = self._benchmark_single_resolution(synth_img, res_name, iterations)
                results.append(bench_res)

        hardware_info = self.cuda_engine.get_hardware_status()

        return {
            "hardware": hardware_info,
            "benchmark_results": results,
            "summary_note": (
                "GPU kernel execution speed depends heavily on pixel parallelism. "
                "For smaller images (e.g. 720p), CPU to GPU memory copy overhead (HtoD / DtoH) "
                "can represent a significant fraction of total GPU latency. For high-resolution images (1080p to 4K), "
                "the massive parallel computing throughput of the RTX 4060 CUDA cores yields massive speedups."
            )
        }

    def _benchmark_single_resolution(self, img: np.ndarray, resolution_label: str, iterations: int = 3) -> dict:
        h, w, c = img.shape
        num_pixels = w * h

        # 1. Warm-up GPU kernels if CUDA is available
        if self.cuda_engine.gpu_available:
            self.cuda_engine.apply_auto_enhance(img)

        operations = ["Gamma Correction", "Brightness & Contrast", "Gaussian Blur", "Sharpening", "Combined Auto Enhance"]
        op_data = []

        for op in operations:
            cpu_times = []
            gpu_kernel_times = []
            gpu_htod_times = []
            gpu_dtoh_times = []
            gpu_total_times = []

            for _ in range(iterations):
                # Measure CPU Execution
                t0_cpu = time.perf_counter()
                if op == "Gamma Correction":
                    _ = CPUEngine.apply_gamma(img, 2.0)
                elif op == "Brightness & Contrast":
                    _ = CPUEngine.apply_brightness_contrast(img, 15.0, 1.3)
                elif op == "Gaussian Blur":
                    _ = CPUEngine.apply_gaussian_blur(img, 5)
                elif op == "Sharpening":
                    _ = CPUEngine.apply_sharpen(img, 1.0)
                elif op == "Combined Auto Enhance":
                    _ = CPUEngine.apply_auto_enhance(img, gamma=2.0, brightness=15.0, contrast=1.3, denoising_strength=0.5, sharpening_strength=0.5)
                cpu_time_ms = (time.perf_counter() - t0_cpu) * 1000.0
                cpu_times.append(cpu_time_ms)

                # Measure CUDA GPU Execution
                if self.cuda_engine.gpu_available:
                    if op == "Gamma Correction":
                        _, m = self.cuda_engine.apply_gamma(img, 2.0)
                    elif op == "Brightness & Contrast":
                        _, m = self.cuda_engine.apply_brightness_contrast(img, 15.0, 1.3)
                    elif op == "Gaussian Blur":
                        _, m = self.cuda_engine.apply_gaussian_blur(img, 5, use_shared_memory=True)
                    elif op == "Sharpening":
                        _, m = self.cuda_engine.apply_sharpen(img, 1.0)
                    elif op == "Combined Auto Enhance":
                        _, m = self.cuda_engine.apply_auto_enhance(img, gamma=2.0, brightness=15.0, contrast=1.3, denoising_strength=0.5, sharpening_strength=0.5)

                    gpu_kernel_times.append(m["kernel_time_ms"])
                    gpu_htod_times.append(m["htod_ms"])
                    gpu_dtoh_times.append(m["dtoh_ms"])
                    gpu_total_times.append(m["total_time_ms"])

            avg_cpu = float(np.mean(cpu_times))
            if self.cuda_engine.gpu_available:
                avg_gpu_kernel = float(np.mean(gpu_kernel_times))
                avg_gpu_htod = float(np.mean(gpu_htod_times))
                avg_gpu_dtoh = float(np.mean(gpu_dtoh_times))
                avg_gpu_total = float(np.mean(gpu_total_times))

                kernel_speedup = avg_cpu / avg_gpu_kernel if avg_gpu_kernel > 0 else 1.0
                e2e_speedup = avg_cpu / avg_gpu_total if avg_gpu_total > 0 else 1.0
            else:
                avg_gpu_kernel = avg_cpu
                avg_gpu_htod = 0.0
                avg_gpu_dtoh = 0.0
                avg_gpu_total = avg_cpu
                kernel_speedup = 1.0
                e2e_speedup = 1.0

            op_data.append({
                "operation": op,
                "cpu_time_ms": round(avg_cpu, 3),
                "gpu_kernel_ms": round(avg_gpu_kernel, 3),
                "gpu_htod_ms": round(avg_gpu_htod, 3),
                "gpu_dtoh_ms": round(avg_gpu_dtoh, 3),
                "gpu_total_ms": round(avg_gpu_total, 3),
                "kernel_speedup": round(kernel_speedup, 2),
                "e2e_speedup": round(e2e_speedup, 2)
            })

        return {
            "resolution": resolution_label,
            "dimensions": f"{w}x{h}",
            "pixel_count": num_pixels,
            "operations": op_data
        }
