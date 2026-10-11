import os
import time
import numpy as np

class CUDAEngine:
    """
    CUDA GPU Acceleration Engine for NightVision AI.
    Loads, compiles, launches, and timings genuine CUDA C++ kernels on NVIDIA GPUs via CuPy / NVRTC.
    """

    def __init__(self, kernels_dir: str = None):
        self.gpu_available = False
        self.device_name = "CPU (CUDA Unavailable)"
        self.vram_total_mb = 0
        self.vram_free_mb = 0
        self.kernels = {}
        
        if kernels_dir is None:
            kernels_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cuda_kernels")
        self.kernels_dir = kernels_dir

        self._initialize_cuda()

    def _initialize_cuda(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                self.gpu_available = True
                props = cp.cuda.runtime.getDeviceProperties(0)
                self.device_name = props['name'].decode('utf-8')
                
                mem_info = cp.cuda.runtime.memGetInfo()
                self.vram_free_mb = round(mem_info[0] / (1024 * 1024), 2)
                self.vram_total_mb = round(mem_info[1] / (1024 * 1024), 2)

                print(f"[CUDAEngine] Initialized on GPU: {self.device_name} (VRAM: {self.vram_free_mb}/{self.vram_total_mb} MB free)")
                self._load_and_compile_kernels()
            else:
                print("[CUDAEngine] No CUDA GPU devices detected. CPU fallback active.")
        except Exception as e:
            print(f"[CUDAEngine] CUDA Initialization warning: {e}. Running in CPU fallback mode.")
            self.gpu_available = False

    def _load_and_compile_kernels(self):
        import cupy as cp
        kernel_files = {
            "gamma": ("gamma.cu", "gamma_correction_kernel"),
            "brightness_contrast": ("brightness_contrast.cu", "brightness_contrast_kernel"),
            "gaussian_global": ("gaussian_blur.cu", "gaussian_blur_global"),
            "gaussian_shared": ("gaussian_blur.cu", "gaussian_blur_shared"),
            "sharpen": ("sharpen.cu", "sharpen_kernel"),
            "combined": ("combined.cu", "combined_auto_enhance_v2"),
        }

        for name, (file_name, kernel_func) in kernel_files.items():
            file_path = os.path.join(self.kernels_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    cuda_code = f.read()
                try:
                    self.kernels[name] = cp.RawKernel(cuda_code, kernel_func)
                    print(f"[CUDAEngine] Compiled kernel '{name}' ({kernel_func}) successfully.")
                except Exception as err:
                    print(f"[CUDAEngine] Error compiling CUDA kernel '{name}': {err}")

    def get_hardware_status(self) -> dict:
        return {
            "gpu_available": self.gpu_available,
            "device_name": self.device_name,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
            "backend": "CUDA GPU" if self.gpu_available else "CPU (Fallback)",
            "compiled_kernels": list(self.kernels.keys())
        }

    def apply_gamma(self, img: np.ndarray, gamma: float) -> tuple[np.ndarray, dict]:
        """Runs Gamma Correction CUDA C++ kernel on GPU with precise CUDA timing."""
        if not self.gpu_available or "gamma" not in self.kernels:
            from app.cpu_engine import CPUEngine
            t0 = time.perf_counter()
            res = CPUEngine.apply_gamma(img, gamma)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return res, {"kernel_time_ms": duration_ms, "htod_ms": 0.0, "dtoh_ms": 0.0, "total_time_ms": duration_ms, "backend": "CPU (Fallback)"}

        import cupy as cp
        height, width, channels = img.shape

        t_start_e2e = time.perf_counter()
        
        # 1. HtoD Transfer
        t0_htod = time.perf_counter()
        d_src = cp.asarray(img)
        d_dst = cp.empty_like(d_src)
        cp.cuda.Stream.null.synchronize()
        htod_ms = (time.perf_counter() - t0_htod) * 1000.0

        # Grid configuration
        threadsperblock = (16, 16)
        blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # 2. Kernel Execution with CUDA Events
        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()

        start_evt.record()
        self.kernels["gamma"](
            blockspergrid, threadsperblock,
            (d_src, d_dst, width, height, channels, np.float32(gamma))
        )
        end_evt.record()
        end_evt.synchronize()

        kernel_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))

        # 3. DtoH Transfer
        t0_dtoh = time.perf_counter()
        result_img = cp.asnumpy(d_dst)
        dtoh_ms = (time.perf_counter() - t0_dtoh) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_start_e2e) * 1000.0

        metrics = {
            "kernel_time_ms": round(kernel_ms, 4),
            "htod_ms": round(htod_ms, 4),
            "dtoh_ms": round(dtoh_ms, 4),
            "total_time_ms": round(total_e2e_ms, 4),
            "backend": f"CUDA ({self.device_name})"
        }

        return result_img, metrics

    def apply_brightness_contrast(self, img: np.ndarray, brightness: float, contrast: float) -> tuple[np.ndarray, dict]:
        """Runs Brightness and Contrast CUDA C++ kernel."""
        if not self.gpu_available or "brightness_contrast" not in self.kernels:
            from app.cpu_engine import CPUEngine
            t0 = time.perf_counter()
            res = CPUEngine.apply_brightness_contrast(img, brightness, contrast)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return res, {"kernel_time_ms": duration_ms, "htod_ms": 0.0, "dtoh_ms": 0.0, "total_time_ms": duration_ms, "backend": "CPU (Fallback)"}

        import cupy as cp
        height, width, channels = img.shape

        t_start_e2e = time.perf_counter()
        t0_htod = time.perf_counter()
        d_src = cp.asarray(img)
        d_dst = cp.empty_like(d_src)
        cp.cuda.Stream.null.synchronize()
        htod_ms = (time.perf_counter() - t0_htod) * 1000.0

        threadsperblock = (16, 16)
        blockspergrid = ((width + 15) // 16, (height + 15) // 16)

        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()

        start_evt.record()
        self.kernels["brightness_contrast"](
            blockspergrid, threadsperblock,
            (d_src, d_dst, width, height, channels, np.float32(brightness), np.float32(contrast))
        )
        end_evt.record()
        end_evt.synchronize()

        kernel_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))

        t0_dtoh = time.perf_counter()
        result_img = cp.asnumpy(d_dst)
        dtoh_ms = (time.perf_counter() - t0_dtoh) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_start_e2e) * 1000.0

        return result_img, {
            "kernel_time_ms": round(kernel_ms, 4),
            "htod_ms": round(htod_ms, 4),
            "dtoh_ms": round(dtoh_ms, 4),
            "total_time_ms": round(total_e2e_ms, 4),
            "backend": f"CUDA ({self.device_name})"
        }

    def apply_gaussian_blur(self, img: np.ndarray, ksize: int = 5, use_shared_memory: bool = True) -> tuple[np.ndarray, dict]:
        """Runs Gaussian Blur CUDA kernel (supporting Shared memory vs Global memory variants)."""
        kernel_name = "gaussian_shared" if use_shared_memory and "gaussian_shared" in self.kernels else "gaussian_global"
        
        if not self.gpu_available or kernel_name not in self.kernels:
            from app.cpu_engine import CPUEngine
            t0 = time.perf_counter()
            res = CPUEngine.apply_gaussian_blur(img, ksize=ksize)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return res, {"kernel_time_ms": duration_ms, "htod_ms": 0.0, "dtoh_ms": 0.0, "total_time_ms": duration_ms, "backend": "CPU (Fallback)"}

        import cupy as cp
        height, width, channels = img.shape

        # 5x5 Gaussian Kernel weights
        k5x5 = np.array([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1]
        ], dtype=np.float32) / 273.0

        t_start_e2e = time.perf_counter()
        t0_htod = time.perf_counter()
        d_src = cp.asarray(img)
        d_dst = cp.empty_like(d_src)
        d_kernel = cp.asarray(k5x5)
        cp.cuda.Stream.null.synchronize()
        htod_ms = (time.perf_counter() - t0_htod) * 1000.0

        threadsperblock = (16, 16)
        blockspergrid = ((width + 15) // 16, (height + 15) // 16)

        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()

        start_evt.record()
        self.kernels[kernel_name](
            blockspergrid, threadsperblock,
            (d_src, d_dst, width, height, channels, d_kernel, 5)
        )
        end_evt.record()
        end_evt.synchronize()

        kernel_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))

        t0_dtoh = time.perf_counter()
        result_img = cp.asnumpy(d_dst)
        dtoh_ms = (time.perf_counter() - t0_dtoh) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_start_e2e) * 1000.0

        return result_img, {
            "kernel_time_ms": round(kernel_ms, 4),
            "htod_ms": round(htod_ms, 4),
            "dtoh_ms": round(dtoh_ms, 4),
            "total_time_ms": round(total_e2e_ms, 4),
            "backend": f"CUDA ({kernel_name})"
        }

    def apply_sharpen(self, img: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, dict]:
        """Runs Sharpening CUDA C++ kernel."""
        if not self.gpu_available or "sharpen" not in self.kernels:
            from app.cpu_engine import CPUEngine
            t0 = time.perf_counter()
            res = CPUEngine.apply_sharpen(img, strength=strength)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return res, {"kernel_time_ms": duration_ms, "htod_ms": 0.0, "dtoh_ms": 0.0, "total_time_ms": duration_ms, "backend": "CPU (Fallback)"}

        import cupy as cp
        height, width, channels = img.shape

        t_start_e2e = time.perf_counter()
        t0_htod = time.perf_counter()
        d_src = cp.asarray(img)
        d_dst = cp.empty_like(d_src)
        cp.cuda.Stream.null.synchronize()
        htod_ms = (time.perf_counter() - t0_htod) * 1000.0

        threadsperblock = (16, 16)
        blockspergrid = ((width + 15) // 16, (height + 15) // 16)

        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()

        start_evt.record()
        self.kernels["sharpen"](
            blockspergrid, threadsperblock,
            (d_src, d_dst, width, height, channels, np.float32(strength))
        )
        end_evt.record()
        end_evt.synchronize()

        kernel_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))

        t0_dtoh = time.perf_counter()
        result_img = cp.asnumpy(d_dst)
        dtoh_ms = (time.perf_counter() - t0_dtoh) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_start_e2e) * 1000.0

        return result_img, {
            "kernel_time_ms": round(kernel_ms, 4),
            "htod_ms": round(htod_ms, 4),
            "dtoh_ms": round(dtoh_ms, 4),
            "total_time_ms": round(total_e2e_ms, 4),
            "backend": f"CUDA ({self.device_name})"
        }

    def apply_auto_enhance(
        self,
        img: np.ndarray,
        gamma: float = 2.0,
        brightness: float = 10.0,
        contrast: float = 1.25,
        denoising_strength: float = 0.5,
        sharpening_strength: float = 0.5
    ) -> tuple[np.ndarray, dict]:
        """Runs single-pass Combined Auto Enhance CUDA C++ kernel."""
        if not self.gpu_available or "combined" not in self.kernels:
            from app.cpu_engine import CPUEngine
            t0 = time.perf_counter()
            res = CPUEngine.apply_auto_enhance(img, gamma, brightness, contrast, denoising_strength, sharpening_strength)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return res, {"kernel_time_ms": duration_ms, "htod_ms": 0.0, "dtoh_ms": 0.0, "total_time_ms": duration_ms, "backend": "CPU (Fallback)"}

        import cupy as cp
        height, width, channels = img.shape

        t_start_e2e = time.perf_counter()
        t0_htod = time.perf_counter()
        d_src = cp.asarray(img)
        d_dst = cp.empty_like(d_src)
        cp.cuda.Stream.null.synchronize()
        htod_ms = (time.perf_counter() - t0_htod) * 1000.0

        threadsperblock = (16, 16)
        blockspergrid = ((width + 15) // 16, (height + 15) // 16)

        start_evt = cp.cuda.Event()
        end_evt = cp.cuda.Event()

        start_evt.record()
        self.kernels["combined"](
            blockspergrid, threadsperblock,
            (
                d_src, d_dst, width, height, channels,
                np.float32(gamma), np.float32(brightness), np.float32(contrast), np.float32(sharpening_strength)
            )
        )
        end_evt.record()
        end_evt.synchronize()

        kernel_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))

        t0_dtoh = time.perf_counter()
        result_img = cp.asnumpy(d_dst)
        dtoh_ms = (time.perf_counter() - t0_dtoh) * 1000.0

        total_e2e_ms = (time.perf_counter() - t_start_e2e) * 1000.0

        return result_img, {
            "kernel_time_ms": round(kernel_ms, 4),
            "htod_ms": round(htod_ms, 4),
            "dtoh_ms": round(dtoh_ms, 4),
            "total_time_ms": round(total_e2e_ms, 4),
            "backend": f"CUDA ({self.device_name})"
        }
