# NightVision AI — Complete Technical & Implementation Specification

**Project Title**: NightVision AI — GPU-Accelerated Low-Light Photo & Video Enhancement Studio  
**Course / Field**: GPU Programming & High-Performance Parallel Computing / Computer Vision  
**Author / Repository Owner**: Vaibhavi0106  
**Repository**: [https://github.com/Vaibhavi0106/NightVision-AI](https://github.com/Vaibhavi0106/NightVision-AI)  
**Target Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, 8GB GDDR6 VRAM, Driver 591.66, CUDA 13.1)  
**Software Platform**: Python 3.13 (FastAPI, CuPy NVRTC, OpenCV, NumPy) + Node.js (React 18, TypeScript, Vite, Tailwind CSS)

---

## Table of Contents
1. [Executive Summary & Objectives](#1-executive-summary--objectives)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [CUDA Programming & Parallel Compute Architecture](#3-cuda-programming--parallel-compute-architecture)
   - [3.1 Thread-to-Pixel Mapping & Grid Topology](#31-thread-to-pixel-mapping--grid-topology)
   - [3.2 Memory Hierarchy & Transfer Overhead (HtoD / DtoH)](#32-memory-hierarchy--transfer-overhead-htod--dtoh)
   - [3.3 Detailed CUDA C++ Kernel Implementations](#33-detailed-cuda-c-kernel-implementations)
     - [A. Gamma Correction (`gamma.cu`)](#a-gamma-correction-gammacu)
     - [B. Brightness & Contrast Adjustment (`brightness_contrast.cu`)](#b-brightness--contrast-adjustment-brightness_contrastcu)
     - [C. Shared-Memory Tiled Gaussian Blur Convolution (`gaussian_blur.cu`)](#c-shared-memory-tiled-gaussian-blur-convolution-gaussian_blurcu)
     - [D. Laplacian Edge Sharpening (`sharpen.cu`)](#d-laplacian-edge-sharpening-sharpencu)
     - [E. Single-Pass Fused Auto-Enhancement (`combined.cu`)](#e-single-pass-fused-auto-enhancement-combinedcu)
4. [CPU Reference Engine & Mathematical Equivalence](#4-cpu-reference-engine--mathematical-equivalence)
5. [Video Enhancement & Streaming Pipeline](#5-video-enhancement--streaming-pipeline)
6. [Batch Processing Architecture](#6-batch-processing-architecture)
7. [Performance Lab & Benchmarking Analysis](#7-performance-lab--benchmarking-analysis)
8. [Backend REST API Specification](#8-backend-rest-api-specification)
9. [Frontend UI/UX & State Architecture](#9-frontend-uiux--state-architecture)
10. [Numerical Verification & Automated Testing](#10-numerical-verification--automated-testing)
11. [Deployment, Packaging & Reproducibility](#11-deployment-packaging--reproducibility)

---

## 1. Executive Summary & Objectives

NightVision AI is an end-to-end, high-performance photo and video enhancement studio designed specifically for GPU parallel computing demonstration and real-world deployment. The system addresses the classical computer vision problem of underexposed and low-light imagery: degraded visibility, heavy sensor noise, loss of dynamic range, and edge blunting.

### Core Engineering Requirements:
1. **Genuine GPU Acceleration**: Custom CUDA C++ kernels compiled at runtime via NVIDIA NVRTC (CUDA Runtime Compiler) through CuPy, executing on genuine GPU hardware rather than emulated wrappers.
2. **Deterministic Mathematical Parity**: Strict numerical equivalence between custom CUDA C++ kernels and standard CPU implementations (Mean Absolute Error $\text{MAE} < 1.0$ across dynamic range $[0, 255]$).
3. **Graceful CPU Fallback**: Dynamic hardware detection allowing full software execution on devices without NVIDIA GPUs without application crashes.
4. **Memory-Conscious Video Processing**: Frame-by-frame streaming processing to prevent GPU/Host out-of-memory (OOM) faults on long videos.
5. **Modern Desktop-Quality Web UX**: Responsive React 18 single-page application built with Tailwind CSS, Recharts for performance visualization, split-screen interactive comparison sliders, and real-time backend hardware telemetry.

---

## 2. End-to-End System Architecture

The project is structured as a decoupled client-server architecture with an optional unified production deployment mode:

```
+---------------------------------------------------------------------------------------+
|                                    CLIENT BROWSER                                     |
|  React 18 + TypeScript + Vite + Tailwind CSS (Pink & White High-Contrast Theme)        |
|  - Dashboard: System Telemetry & Quick Enhancer                                       |
|  - Image Enhancer: Interactive Split-View & Custom Sliders                            |
|  - Video Enhancer: Async Polling & Frame Progress Display                              |
|  - Batch Processing: Multi-File Queue & Zip Downloader                                |
|  - Performance Lab: Multi-Resolution CPU vs CUDA Benchmarking (Recharts)              |
+------------------------------------------+--------------------------------------------+
                                           | HTTP REST API (JSON / Multipart Form-Data)
                                           v
+---------------------------------------------------------------------------------------+
|                                 FASTAPI BACKEND SERVER                                |
|  Uvicorn ASGI Server / Python 3.13 Runtime                                            |
|  - /api/health: GPU Telemetry (Device, VRAM, Active Kernels)                          |
|  - /api/image/enhance: Single Image Processing Endpoint                               |
|  - /api/video/enhance: Background Threaded Video Streaming Pipeline                   |
|  - /api/batch/process: Concurrent Batch Processing & Zip Packaging                     |
|  - /api/benchmark/run: Performance Lab Automated Profiler                            |
+------------------------------------------+--------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+     +---------------------------------------+
|           CUDA GPU ENGINE             |     |           CPU FALLBACK ENGINE         |
|  CuPy 14.2 + NVRTC C++ Kernels        |     |  NumPy Vectorization + OpenCV         |
|  - Raw CUDA C++ Compilation           |     |  - 8-bit Precomputed LUT Gamma        |
|  - 2D Grid Block Allocation           |     |  - Vectorized Contrast & Brightness   |
|  - Shared-Memory 2D Tiled Stencil     |     |  - YCrCb Separable CLAHE              |
|  - Fused Single-Pass Kernel           |     |  - CPU Gaussian & Median Filters      |
|  - CUDA Event Micro-Benchmarking      |     |  - 3x3 Laplacian Convolution Filter   |
+---------------------------------------+     +---------------------------------------+
```

---

## 3. CUDA Programming & Parallel Compute Architecture

### 3.1 Thread-to-Pixel Mapping & Grid Topology

In digital image processing on the GPU, a 2D image matrix of size $W \times H$ with $C$ color channels (typically interleaved BGR or RGB) is mapped onto a two-dimensional grid of thread blocks:

- **Block Dimensions**: Each thread block is configured as a square tile:
  $$\text{blockDim.x} = 16, \quad \text{blockDim.y} = 16 \implies 256 \text{ threads per block}$$
  *Rationale*: A block size of 256 threads matches the warp size of 32 (8 warps per block), ensuring high multiprocessor occupancy on NVIDIA Ada Lovelace architectures (RTX 4060) while maintaining balanced register pressure.

- **Grid Dimensions**: The number of blocks required to cover the image canvas:
  $$\text{gridDim.x} = \left\lceil \frac{W}{\text{blockDim.x}} \right\rceil = \left\lfloor \frac{W + 15}{16} \right\rfloor$$
  $$\text{gridDim.y} = \left\lceil \frac{H}{\text{blockDim.y}} \right\rceil = \left\lfloor \frac{H + 15}{16} \right\rfloor$$

- **Global Coordinate Computation & Boundary Guards**:
  Inside every kernel, threads compute their global $(x, y)$ coordinate:
  $$x = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
  $$y = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
  
  Since image dimensions are not necessarily exact multiples of 16, an explicit boundary check prevents out-of-bounds illegal memory access:
  ```cpp
  if (x < width && y < height) {
      // Safe execution inside valid image matrix
  }
  ```

---

### 3.2 Memory Hierarchy & Transfer Overhead (HtoD / DtoH)

A key concept demonstrated in this project is the impact of PCIe bus data transfer overhead versus raw GPU computation time:

1. **Host-to-Device (HtoD)**: Transfer of raw pixel data from host RAM to GPU device memory using `cp.asarray(img)`.
2. **Device Kernel Execution**: Raw arithmetic operations executed in parallel by Streaming Multiprocessors (SMs).
3. **Device-to-Host (DtoH)**: Transfer of modified pixel buffers back to CPU memory using `cp.asnumpy(d_dst)`.

#### CUDA Event Micro-Benchmarking
To measure pure GPU kernel compute time without including PCIe latency or driver overhead, high-precision CUDA Events are utilized:
```python
start_evt = cp.cuda.Event()
end_evt = cp.cuda.Event()

start_evt.record()
self.kernels["gamma"](blockspergrid, threadsperblock, (d_src, d_dst, width, height, channels, np.float32(gamma)))
end_evt.record()
end_evt.synchronize()

kernel_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))
```

---

### 3.3 Detailed CUDA C++ Kernel Implementations

#### A. Gamma Correction (`gamma.cu`)
Low-light imagery compresses shadow details into the lower range of the sensor. Non-linear power-law gamma transformation redistributes luminance non-linearly:

$$S = \left(\frac{I}{255}\right)^{\frac{1}{\gamma}} \times 255$$

Where:
- $\gamma > 1.0$ lifts dark shadows while non-linearly preserving highlight integrity.
- `__powf()` is an intrinsic fast single-precision hardware math instruction on NVIDIA GPUs.

```cpp
extern "C" __global__
void gamma_correction_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float gamma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * channels;
        float inv_gamma = 1.0f / gamma;

        for (int c = 0; c < channels; ++c) {
            float norm = (float)src[pixel_idx + c] / 255.0f;
            float corrected = __powf(norm, inv_gamma);
            int val = (int)(corrected * 255.0f + 0.5f);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            dst[pixel_idx + c] = (unsigned char)val;
        }
    }
}
```

---

#### B. Brightness & Contrast Adjustment (`brightness_contrast.cu`)
Applies linear contrast scaling centered at mid-gray ($128$) and adds uniform brightness:

$$O = \alpha \cdot (I - 128) + 128 + \beta$$

Where $\alpha$ is the contrast multiplier and $\beta$ is the brightness offset. Safe clamping ensures values remain within $[0, 255]$.

```cpp
extern "C" __global__
void brightness_contrast_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float brightness,
    float contrast
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * channels;

        for (int c = 0; c < channels; ++c) {
            float val = (float)src[pixel_idx + c];
            float res = contrast * (val - 128.0f) + 128.0f + brightness;
            int out_val = (int)(res + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;
            dst[pixel_idx + c] = (unsigned char)out_val;
        }
    }
}
```

---

#### C. Shared-Memory Tiled Gaussian Blur Convolution (`gaussian_blur.cu`)
Low-light enhancement inherently amplifies high-frequency sensor noise. 2D Gaussian convolution dampens noise by weighted spatial averaging.

##### The Shared Memory Advantage:
In a naive global memory convolution, every thread reads a $(2R+1) \times (2R+1)$ window of pixels directly from high-latency device DRAM. For a $5 \times 5$ kernel ($R=2$), each pixel is fetched up to 25 times from global memory.

By loading a tile of size $(16 + 2R) \times (16 + 2R)$ into fast on-chip `__shared__` memory, redundant global memory transactions are minimized:
1. Each thread loads its interior pixel into shared memory.
2. Boundary threads cooperatively load the halo/apron pixels.
3. `__syncthreads()` establishes a barrier synchronization point, ensuring the entire tile is loaded before convolution begins.
4. All 25 stencil lookups execute directly against low-latency shared memory banks.

```cpp
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define MAX_KERNEL_RADIUS 5
#define TILE_WIDTH (BLOCK_DIM_X + 2 * MAX_KERNEL_RADIUS)
#define TILE_HEIGHT (BLOCK_DIM_Y + 2 * MAX_KERNEL_RADIUS)

extern "C" __global__
void gaussian_blur_shared(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    const float* __restrict__ kernel,
    int k_size
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int radius = k_size / 2;

    // Shared memory tile with halo apron
    __shared__ float s_tile[TILE_HEIGHT][TILE_WIDTH][3];

    // Cooperative loading of main pixel and halo borders
    int g_x = min(max(x - radius, 0), width - 1);
    int g_y = min(max(y - radius, 0), height - 1);

    for (int c = 0; c < channels && c < 3; ++c) {
        s_tile[ty][tx][c] = (float)src[(g_y * width + g_x) * channels + c];
    }

    // Load right halo
    if (tx < 2 * radius) {
        int halo_x = min(max(x + blockDim.x - radius, 0), width - 1);
        for (int c = 0; c < channels && c < 3; ++c) {
            s_tile[ty][tx + blockDim.x][c] = (float)src[(g_y * width + halo_x) * channels + c];
        }
    }

    // Load bottom halo
    if (ty < 2 * radius) {
        int halo_y = min(max(y + blockDim.y - radius, 0), height - 1);
        for (int c = 0; c < channels && c < 3; ++c) {
            s_tile[ty + blockDim.y][tx][c] = (float)src[(halo_y * width + g_x) * channels + c];
        }
    }

    // Load bottom-right diagonal corner
    if (tx < 2 * radius && ty < 2 * radius) {
        int halo_x = min(max(x + blockDim.x - radius, 0), width - 1);
        int halo_y = min(max(y + blockDim.y - radius, 0), height - 1);
        for (int c = 0; c < channels && c < 3; ++c) {
            s_tile[ty + blockDim.y][tx + blockDim.x][c] = (float)src[(halo_y * width + halo_x) * channels + c];
        }
    }

    __syncthreads(); // Barrier: wait for entire shared tile

    if (x < width && y < height) {
        for (int c = 0; c < channels && c < 3; ++c) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int ky = 0; ky < k_size; ++ky) {
                for (int kx = 0; kx < k_size; ++kx) {
                    float w = kernel[ky * k_size + kx];
                    sum += s_tile[ty + ky][tx + kx][c] * w;
                    weight_sum += w;
                }
            }

            int out_val = (int)(sum / weight_sum + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;
            dst[(y * width + x) * channels + c] = (unsigned char)out_val;
        }
    }
}
```

---

#### D. Laplacian Edge Sharpening (`sharpen.cu`)
Enhances visual acuity after denoising by performing a 2D discrete Laplacian convolution:

$$\nabla^2 I(x, y) = 4 \cdot I(x, y) - \left[ I(x+1, y) + I(x-1, y) + I(x, y+1) + I(x, y-1) \right]$$
$$I_{\text{sharp}}(x, y) = I(x, y) + \lambda \cdot \nabla^2 I(x, y)$$

Where $\lambda$ controls the sharpening intensity.

```cpp
extern "C" __global__
void sharpen_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float center = (float)src[(y * width + x) * channels + c];

            int top_y    = max(y - 1, 0);
            int bottom_y = min(y + 1, height - 1);
            int left_x   = max(x - 1, 0);
            int right_x  = min(x + 1, width - 1);

            float top    = (float)src[(top_y * width + x) * channels + c];
            float bottom = (float)src[(bottom_y * width + x) * channels + c];
            float left   = (float)src[(y * width + left_x) * channels + c];
            float right  = (float)src[(y * width + right_x) * channels + c];

            float laplacian = 4.0f * center - (top + bottom + left + right);
            float enhanced = center + strength * laplacian;

            int out_val = (int)(enhanced + 0.5f);
            if (out_val < 0) out_val = 0;
            if (out_val > 255) out_val = 255;

            dst[(y * width + x) * channels + c] = (unsigned char)out_val;
        }
    }
}
```

---

#### E. Single-Pass Fused Auto-Enhancement (`combined.cu`)
A major performance bottleneck in multi-stage image processing pipelines is **memory bandwidth saturation**. If an application executes Gamma Correction, Brightness Adjustment, and Sharpening as three distinct kernels:
- Intermediate results must be written back to high-bandwidth memory (DRAM) and re-read in subsequent passes.
- 3 kernel launch calls incur driver overhead.

**NightVision AI's Fused Kernel Solution**:
The `combined_auto_enhance_v2` kernel fuses Gamma, Brightness, Contrast, and Laplacian Sharpening into a **single GPU invocation**. Intermediate values are stored directly inside fast GPU register memory, eliminating DRAM round-trips.

```cpp
extern "C" __global__
void combined_auto_enhance_v2(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    int channels,
    float gamma,
    float brightness,
    float contrast,
    float sharpen_strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float inv_gamma = 1.0f / gamma;
        int center_idx = (y * width + x) * channels;

        for (int c = 0; c < channels; ++c) {
            #define TRANSFORM_AND_CLAMP(idx) \
                fminf(fmaxf(contrast * ((__powf((float)src[(idx) + c] / 255.0f, inv_gamma) * 255.0f) - 128.0f) + 128.0f + brightness, 0.0f), 255.0f)

            float center_cb = TRANSFORM_AND_CLAMP(center_idx);

            float final_val = center_cb;
            if (sharpen_strength > 0.001f) {
                int top_y    = max(y - 1, 0);
                int bottom_y = min(y + 1, height - 1);
                int left_x   = max(x - 1, 0);
                int right_x  = min(x + 1, width - 1);

                int top_idx    = (top_y * width + x) * channels;
                int bottom_idx = (bottom_y * width + x) * channels;
                int left_idx   = (y * width + left_x) * channels;
                int right_idx  = (y * width + right_x) * channels;

                float top_cb    = TRANSFORM_AND_CLAMP(top_idx);
                float bottom_cb = TRANSFORM_AND_CLAMP(bottom_idx);
                float left_cb   = TRANSFORM_AND_CLAMP(left_idx);
                float right_cb  = TRANSFORM_AND_CLAMP(right_idx);

                float lap = 4.0f * center_cb - (top_cb + bottom_cb + left_cb + right_cb);
                final_val = center_cb + sharpen_strength * lap;
            }

            #undef TRANSFORM_AND_CLAMP

            int out_int = (int)(final_val + 0.5f);
            if (out_int < 0) out_int = 0;
            if (out_int > 255) out_int = 255;

            dst[center_idx + c] = (unsigned char)out_int;
        }
    }
}
```

---

## 4. CPU Reference Engine & Mathematical Equivalence

To fulfill academic rigor and ensure systems without an NVIDIA GPU function seamlessly, `backend/app/cpu_engine.py` implements CPU reference functions:

1. **8-bit Lookup Table (LUT) Gamma Correction**:
   ```python
   table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
   return cv2.LUT(img, table)
   ```
2. **Auto-Brightness Estimation**: Computes mean luminance in gray-space and dynamically calculates optimal gamma and contrast:
   $$\text{target\_lum} = 120.0, \quad \gamma_{\text{calc}} = \text{clip}\left(\frac{\ln(120 / 255)}{\ln(\text{mean\_lum} / 255)}, 1.2, 3.5\right)$$
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
   Applied in the $Y\text{Cr}\text{Cb}$ color space on the luminance channel ($Y$) only, preserving chrominance ($Cr, Cb$) without color distortion:
   ```python
   ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
   y, cr, cb = cv2.split(ycrcb)
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   y_eq = clahe.apply(y)
   return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
   ```

---

## 5. Video Enhancement & Streaming Pipeline

Processing high-definition video requires careful memory management to prevent memory exhaustion:
- Loading an entire 4K video into RAM or VRAM triggers Out-Of-Memory (OOM) failures.
- **NightVision AI Architecture**: Uses a generator-style sequential frame-by-frame streaming loop.

```
+-----------------------------------------------------------------------------------------+
|                                VIDEO PROCESSING PIPELINE                                |
+-----------------------------------------------------------------------------------------+
                                             |
                                  1. Open VideoCapture
                                             v
                              2. Read Single Frame [H, W, 3]
                                             v
               +-----------------------------+-----------------------------+
               | (If CUDA GPU Available)                                   | (If CPU Fallback)
               v                                                           v
       Transfer to GPU VRAM                                         Direct CPU Enhancement
       Execute Fused CUDA Kernel                                    Apply Vectorized Operations
       Retrieve Enhanced Frame                                             |
               |                                                           |
               +-----------------------------> <---------------------------+
                                             |
                                3. Write via VideoWriter
                                             v
                               4. Update Job Progress & FPS
                                             v
                              (Repeat until EOF is reached)
                                             v
                                  5. Release Handles
                                             v
                        6. Optional FFmpeg Audio Muxing Pass
                                             v
                                   7. Mark Completed
```

### Video Job Status Polling
Video enhancement runs inside a detached daemon thread. The frontend polls `/api/video/status/{job_id}` at 500ms intervals to update:
- Processed frames / Total frames
- Real-time processing speed (Frames Per Second)
- Progress percentage bar
- Side-by-side video playback comparison upon completion

---

## 6. Batch Processing Architecture

The Batch Processing module allows multiple images to be dropped into a queue:
- **Independent Execution**: Images are processed sequentially to avoid GPU context contention.
- **Zip Archiving**: Enhanced images are saved into a temporary directory and packaged into a single downloadable `.zip` archive on the fly using Python's `zipfile` library.
- **Fail-Safe Tolerance**: If one image format is corrupt, the server records the error on that item while continuing processing for the remaining queue.

---

## 7. Performance Lab & Benchmarking Analysis

The Performance Lab module (`PerformanceBenchmarker`) performs automated side-by-side benchmarking between the CPU and CUDA GPU across standard video/display resolutions:
- **720p**: $1280 \times 720$ (0.92 Million Pixels)
- **1080p Full HD**: $1920 \times 1080$ (2.07 Million Pixels)
- **1440p 2K**: $2560 \times 1440$ (3.68 Million Pixels)
- **4K Ultra HD**: $3840 \times 2160$ (8.29 Million Pixels)

### Performance Insights & Amdahl's Law
1. **Low Resolutions (720p)**: PCIe host-to-device copy overhead represents ~40–60% of total elapsed latency. The compute workload is not large enough to completely hide PCIe bus overhead.
2. **High Resolutions (1080p, 4K)**: The massive parallel compute capacity of the RTX 4060 GPU dominates. In operations such as 5x5 convolution and fused auto-enhancement, the CUDA kernel achieves **up to 15x–35x speedups** over single-threaded CPU processing.

---

## 8. Backend REST API Specification

| Endpoint | Method | Input Parameters | Output Response | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | None | `{status, hardware, stats}` | Hardware detection, VRAM metrics, and session statistics. |
| `/api/files/{id}` | `GET` | `file_id: str` | Binary file stream | Serves images, videos, and generated zip archives. |
| `/api/image/upload` | `POST` | `file: UploadFile` | `{file_id, width, height, preview_url}` | Validates and stores source low-light image. |
| `/api/image/sample` | `GET` | None | `{file_id, width, height, preview_url}` | Loads built-in underexposed sample test image. |
| `/api/image/enhance` | `POST` | `file_id, gamma, brightness, contrast, denoising, sharpening, use_clahe, auto_enhance, backend` | `{enhanced_url, metrics}` | Enhances image; returns processing metrics and latency breakdown. |
| `/api/video/upload` | `POST` | `file: UploadFile` | `{file_id, width, height, fps, total_frames, duration_sec}` | Validates video container and extracts stream metadata. |
| `/api/video/enhance` | `POST` | `file_id, gamma, brightness, contrast, denoising, sharpening, backend` | `{job_id, status}` | Initiates asynchronous video frame processing job. |
| `/api/video/status/{id}` | `GET` | `job_id: str` | `{status, progress_percentage, processing_fps, result_url}` | Polls progress and real-time processing FPS. |
| `/api/batch/process` | `POST` | `files: list[UploadFile], parameters...` | `{results, zip_url, total_files}` | Processes image queue and bundles into zip. |
| `/api/benchmark/run` | `POST` | Optional `file: UploadFile` | `{benchmark_results, summary_note}` | Runs multi-resolution benchmark suite across CPU vs CUDA. |

---

## 9. Frontend UI/UX & State Architecture

The frontend is built with React 18, Vite, TypeScript, and Tailwind CSS.

### Design System: High-Contrast Pink & White Theme
- **Color Palette**: Clean white background (`#FFFFFF`), pure crisp white cards (`#FFFFFF`), bold hot pink primary accents (`#EC4899`, `#DB2777`), deep black text (`#000000`, `#0F172A`) for complete legibility.
- **Component Breakdown**:
  1. `Navbar.tsx`: Responsive navigation with active route highlighting and live hardware pill.
  2. `BackendStatusBadge.tsx`: Displays active GPU device, VRAM allocation, and fallback status.
  3. `ImageComparison.tsx`: Interactive before/after split slider and side-by-side toggle.
  4. `MetricsCard.tsx`: Displays kernel execution time, PCIe transfer times, and compute backend.
  5. `Dashboard.tsx`: Session overview, recent jobs, quick-action dropzones.
  6. `ImageEnhancer.tsx`: Fine-grained enhancement controls, debounced adjustments, diagnostic metrics.
  7. `VideoEnhancer.tsx`: Real-time frame processing monitor and dual-player preview.
  8. `BatchProcessing.tsx`: Queue manager with per-item progress indicators.
  9. `PerformanceLab.tsx`: Interactive Recharts bar graphs comparing CPU vs CUDA latencies.
  10. `AboutProject.tsx`: In-browser technical documentation and CUDA block diagram references.

---

## 10. Numerical Verification & Automated Testing

To ensure that the CUDA kernels are mathematically sound and produce equivalent output to the CPU reference implementation, `backend/test_backend.py` runs automated unit tests:

$$\text{MAE} = \frac{1}{W \cdot H \cdot C} \sum_{x=1}^{W} \sum_{y=1}^{H} \sum_{c=1}^{C} \left| I_{\text{CUDA}}(x, y, c) - I_{\text{CPU}}(x, y, c) \right|$$

### Verification Results:
- **Brightness & Contrast**: $\text{MAE} = 0.4352$ (within tolerance $\le 1.0$)
- **Gamma Correction**: $\text{MAE} = 0.4866$ (within tolerance $\le 1.0$)
- **Laplacian Sharpening**: $\text{MAE} = 0.1029$ (within tolerance $\le 1.0$)
- **Combined Auto Enhance**: $\text{MAE} = 0.8820$ (within tolerance $\le 1.0$)

---

## 11. Deployment, Packaging & Reproducibility

### A. Quick Start via Production Server
The unified production script builds the frontend and serves both the API and the static React UI on a single port (`http://localhost:8000`):
```bash
python start_production.py
```
Or double-click `deploy.bat` on Windows.

### B. Development Mode
Run backend and frontend concurrently:
```bash
# Terminal 1: Backend
cd backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### C. Docker Deployment
```bash
docker-compose up --build
```
*(Requires NVIDIA Container Toolkit installed for GPU pass-through).*
