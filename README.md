# NightVision AI — GPU-Accelerated Low-Light Photo & Video Enhancement Studio

[![CUDA](https://img.shields.io/badge/CUDA-13.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646cff.svg)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8.svg)](https://tailwindcss.com/)

**NightVision AI** is a desktop-quality web platform engineered for real-time low-light photo and video enhancement. It utilizes custom CUDA C++ kernels compiled at runtime via NVRTC to perform high-throughput GPU pixel processing on dark or underexposed media.

---

## 🌟 Key Features

1. **Genuine CUDA C++ GPU Acceleration**:
   - Custom CUDA C++ kernels compiled on-the-fly using NVIDIA NVRTC and CuPy.
   - 2D CUDA block/grid thread-to-pixel mapping with boundary safety.
   - Shared-memory (`__shared__`) tiled 2D convolution for 5x5 Gaussian blur with `__syncthreads()` barrier synchronization.
   - Single-pass fused CUDA kernel (`combined.cu`) executing Gamma Correction, Brightness, Contrast, and Sharpening in a single GPU pass to avoid DRAM roundtrips.
   - High-precision GPU latency timing using CUDA Events (`cp.cuda.Event()`).

2. **CPU Fallback Architecture**:
   - Automatic hardware inspection detecting NVIDIA GPU capabilities, VRAM, and driver details.
   - Seamless CPU fallback using NumPy and OpenCV so the platform remains fully functional on systems without a compatible GPU.

3. **6 Complete Modules**:
   - **Dashboard**: Hardware status indicator, drag-and-drop upload zone, sample image launcher, session metrics, recent job queue.
   - **Image Enhancer**: Interactive workspace with split-screen comparison slider, side-by-side view, gamma, brightness, contrast, Gaussian denoising, Laplacian sharpening, CLAHE histogram equalization, and diagnostics panel.
   - **Video Enhancer**: Memory-efficient frame-by-frame video processing pipeline, real-time FPS counter, progress percentage bar, and dual video comparison player.
   - **Batch Processing**: Multi-file drop zone, per-file execution queue, individual output downloads, and ZIP archive export.
   - **Performance Lab**: Live benchmarking suite comparing CPU vs CUDA across 720p, 1080p, 1440p, and 4K resolutions with Recharts bar charts and speedup metrics.
   - **About Project**: Complete academic documentation detailing thread block layouts, memory transfer overheads, and kernel mathematical formulations.

---

## 🏗 Technology Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts.
- **Backend**: Python 3.13, FastAPI, Uvicorn, OpenCV (`cv2`), NumPy, Pillow.
- **GPU Acceleration**: CUDA C++, CuPy 14.2 (NVRTC & CUDA Driver API).
- **Video & File Handling**: OpenCV VideoCapture/VideoWriter, FFmpeg (optional audio muxer).

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ and npm
- (Optional for GPU Mode) NVIDIA GPU with latest drivers (Detected: **NVIDIA GeForce RTX 4060 Laptop GPU**, Driver 591.66)

### 1. Launch Backend API Server
```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```
The FastAPI backend will start at `http://127.0.0.1:8000`.

### 2. Launch Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 🧪 Verification & Unit Tests

To run the automated test suite comparing CUDA GPU and CPU outputs within numerical tolerance (MAE < 1.0):

```bash
cd backend
python test_backend.py
```

Expected output:
```text
[CUDAEngine] Initialized on GPU: NVIDIA GeForce RTX 4060 Laptop GPU
[CUDAEngine] Compiled kernel 'gamma' (gamma_correction_kernel) successfully.
[CUDAEngine] Compiled kernel 'brightness_contrast' (brightness_contrast_kernel) successfully.
[CUDAEngine] Compiled kernel 'gaussian_shared' (gaussian_blur_shared) successfully.
[CUDAEngine] Compiled kernel 'sharpen' (sharpen_kernel) successfully.
[CUDAEngine] Compiled kernel 'combined' (combined_auto_enhance_v2) successfully.
[Test] Brightness/Contrast MAE between CPU and CUDA: 0.4352
[Test] Combined Auto Enhance MAE between CPU and CUDA: 0.8820
[Test] Gamma Correction MAE between CPU and CUDA: 0.4866
[Test] Sharpening MAE between CPU and CUDA: 0.1029
Ran 4 tests in 0.642s - OK
```

---

## 📁 Repository Structure

```text
nightvision-ai/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI REST endpoints & background tasks
│   │   ├── cuda_engine.py     # CUDA GPU Kernel launcher & CUDA Event timing
│   │   ├── cpu_engine.py      # CPU Reference implementations
│   │   ├── enhancer.py        # Unified Image/Video enhancement manager
│   │   └── benchmarker.py     # Performance Lab benchmark suite
│   ├── cuda_kernels/
│   │   ├── gamma.cu           # Gamma correction CUDA C++ kernel
│   │   ├── brightness_contrast.cu # Brightness & Contrast CUDA C++ kernel
│   │   ├── gaussian_blur.cu   # Tiled shared-memory Gaussian blur kernel
│   │   ├── sharpen.cu         # Laplacian sharpening CUDA C++ kernel
│   │   └── combined.cu        # Fused single-pass Auto Enhance CUDA kernel
│   ├── test_backend.py        # Numerical tolerance unit tests (CPU vs CUDA)
│   ├── test_cuda.py           # Quick GPU verification script
│   └── requirements.txt       # Python backend dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # ImageComparison, Navbar, MetricsCard, Badges
│   │   ├── pages/             # Dashboard, ImageEnhancer, VideoEnhancer, etc.
│   │   ├── api/client.ts      # REST API client
│   │   └── types/index.ts     # TypeScript interface definitions
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

---

## 🎓 Academic Documentation & CUDA Design Notes

### Thread-to-Pixel Grid Mapping
Image dimensions `(width, height)` are divided into 16x16 2D CUDA thread blocks (`threadsPerBlock(16, 16)`):
$$\text{blocksPerGrid.x} = \lceil \frac{\text{width}}{16} \rceil, \quad \text{blocksPerGrid.y} = \lceil \frac{\text{height}}{16} \rceil$$

Inside each CUDA kernel, global pixel coordinates are determined via:
$$x = \text{blockIdx.x} \times 16 + \text{threadIdx.x}$$
$$y = \text{blockIdx.y} \times 16 + \text{threadIdx.y}$$

### Shared-Memory Tiling (`__shared__`)
In `gaussian_blur.cu`, a shared memory block tile array `s_tile[TILE_H][TILE_W][3]` is loaded by threads in a block including halo pixels. Coalesced loads and `__syncthreads()` ensure all 5x5 stencil reads hit fast shared memory instead of high-latency global VRAM.

### Speedup vs Memory Transfer Overhead
For low-resolution images (e.g. 720p), host-to-device (HtoD) and device-to-host (DtoH) PCIe transfer overheads represent a significant fraction of total execution time. For high-resolution media (1080p to 4K), raw CUDA kernel compute throughput on the RTX 4060 delivers massive speedups over CPU execution.
