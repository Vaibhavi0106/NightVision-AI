import React from 'react';
import {
  BookOpen,
  Code,
  Layers,
  Zap,
  Terminal,
  Grid,
} from 'lucide-react';
import { HardwareStatus } from '../types';

interface Props {
  hardware?: HardwareStatus;
}

export const AboutProject: React.FC<Props> = ({ hardware }) => {
  return (
    <div className="w-full px-4 sm:px-8 lg:px-12 py-8 flex flex-col gap-8">
      {/* Header */}
      <div className="glass-panel rounded-3xl p-8 border border-pink-200 flex flex-col gap-4 shadow-xl bg-gradient-to-r from-white via-pink-50/50 to-rose-50/30">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-pink-100 border border-pink-200 text-rose-700 text-xs font-semibold w-fit">
          <BookOpen className="w-3.5 h-3.5 text-rose-500" /> GPU Programming Course Project
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          NightVision AI Architecture & CUDA Implementation Document
        </h1>
        <p className="text-slate-600 text-sm leading-relaxed">
          Technical design specifications, parallel thread mapping diagrams, shared-memory optimization, and system architecture for GPU-accelerated low-light image & video enhancement.
        </p>
      </div>

      {/* CUDA Kernels Technical Specification */}
      <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2 pb-3 border-b border-pink-100">
          <Grid className="w-5 h-5 text-rose-500" /> CUDA Thread-to-Pixel Mapping & Grid Sizing
        </h2>

        <div className="flex flex-col gap-4 text-xs text-slate-700 leading-relaxed font-sans">
          <p>
            In NightVision AI, image processing algorithms operate on 2D grids of pixels. CUDA threads are mapped to individual pixel coordinates <code className="font-mono text-rose-600 bg-pink-50 px-1.5 py-0.5 rounded border border-pink-200">(x, y)</code> using a two-dimensional grid and block configuration:
          </p>

          <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 font-mono text-xs text-rose-300 space-y-1 shadow-inner">
            <p className="text-slate-400">// Block dimensions (2D thread block structure)</p>
            <p className="text-pink-300">dim3 threadsPerBlock(16, 16); // 256 threads per CTA block</p>
            <p className="text-slate-400 pt-2">// Grid dimensions dynamically calculated from image dimensions</p>
            <p className="text-pink-300">dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);</p>
            <p className="text-slate-400 pt-2">// Global pixel coordinate computation inside kernel:</p>
            <p className="text-emerald-300">int x = blockIdx.x * blockDim.x + threadIdx.x;</p>
            <p className="text-emerald-300">int y = blockIdx.y * blockDim.y + threadIdx.y;</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div className="bg-pink-50/60 p-4 rounded-xl border border-pink-200 flex flex-col gap-2">
              <h3 className="font-bold text-slate-900 text-xs flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-rose-500" /> Boundary Conditions & Memory Safety
              </h3>
              <p className="text-slate-600">
                To prevent out-of-bounds memory accesses when image dimensions are not exact multiples of 16, every kernel enforces explicit bounds validation:
                <code className="block mt-1 text-rose-600 font-mono">if (x &lt; width && y &lt; height) &#123; ... &#125;</code>
              </p>
            </div>

            <div className="bg-pink-50/60 p-4 rounded-xl border border-pink-200 flex flex-col gap-2">
              <h3 className="font-bold text-slate-900 text-xs flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-rose-500" /> Shared-Memory Tiling (`__shared__`)
              </h3>
              <p className="text-slate-600">
                The Gaussian blur kernel utilizes 2D shared-memory tile buffers <code className="font-mono text-rose-600 bg-white px-1 rounded border border-pink-200">__shared__ float s_tile[TILE_H][TILE_W][3]</code> including apron/halo regions with <code className="font-mono text-rose-600 bg-white px-1 rounded border border-pink-200">__syncthreads()</code> barrier synchronization to maximize L1/Shared cache throughput.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CUDA Kernels Inventory */}
      <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2 pb-3 border-b border-pink-100">
          <Code className="w-5 h-5 text-rose-500" /> Implemented CUDA C++ Kernels Summary
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-pink-50/40 border border-pink-200 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">1. Gamma Correction Kernel</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-pink-100 text-rose-700 border border-pink-200">
                gamma.cu
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Applies non-linear luminance expansion <code className="text-rose-600">S = I^(1/gamma)</code> using native floating-point math units <code className="text-rose-600">__powf()</code> to reveal dark details.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-pink-50/40 border border-pink-200 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">2. Brightness & Contrast Kernel</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-pink-100 text-rose-700 border border-pink-200">
                brightness_contrast.cu
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Scales contrast around mid-tone luminance <code className="text-rose-600">128.0</code> and applies brightness offset with <code className="text-rose-600">fminf/fmaxf</code> clamping.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-pink-50/40 border border-pink-200 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">3. Tiled Gaussian Blur Convolution</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-pink-100 text-rose-700 border border-pink-200">
                gaussian_blur.cu
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Implements 5x5 Gaussian noise filtering using both Global memory and Shared-memory tile halo caching.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-pink-50/40 border border-pink-200 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">4. Single-Pass Combined Kernel</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-pink-100 text-rose-700 border border-pink-200">
                combined.cu
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Fuses Gamma + Brightness + Contrast + Sharpening into a single GPU thread pass, avoiding intermediate DRAM roundtrips.
            </p>
          </div>
        </div>
      </div>

      {/* Tech Stack Specs */}
      <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2 pb-3 border-b border-pink-100">
          <Terminal className="w-5 h-5 text-rose-500" /> Technology Stack Overview
        </h2>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3 rounded-xl bg-pink-50/50 border border-pink-200 flex flex-col gap-1">
            <span className="text-slate-500 font-sans">Frontend</span>
            <span className="font-bold text-slate-900">React + Vite + TS</span>
          </div>

          <div className="p-3 rounded-xl bg-pink-50/50 border border-pink-200 flex flex-col gap-1">
            <span className="text-slate-500 font-sans">Styling</span>
            <span className="font-bold text-slate-900">Tailwind CSS</span>
          </div>

          <div className="p-3 rounded-xl bg-pink-50/50 border border-pink-200 flex flex-col gap-1">
            <span className="text-slate-500 font-sans">Charts</span>
            <span className="font-bold text-slate-900">Recharts</span>
          </div>

          <div className="p-3 rounded-xl bg-pink-50/50 border border-pink-200 flex flex-col gap-1">
            <span className="text-slate-500 font-sans">Backend</span>
            <span className="font-bold text-slate-900">Python FastAPI</span>
          </div>

          <div className="p-3 rounded-xl bg-pink-50/50 border border-pink-200 flex flex-col gap-1">
            <span className="text-slate-500 font-sans">Image / Video</span>
            <span className="font-bold text-slate-900">OpenCV + NumPy</span>
          </div>

          <div className="p-3 rounded-xl bg-pink-50/50 border border-pink-200 flex flex-col gap-1">
            <span className="text-slate-500 font-sans">GPU Acceleration</span>
            <span className="font-bold text-rose-600">CUDA C++ / CuPy</span>
          </div>
        </div>
      </div>
    </div>
  );
};
