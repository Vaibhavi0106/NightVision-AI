import React, { useRef } from 'react';
import {
  Upload,
  Sparkles,
  Zap,
  Image as ImageIcon,
  Film,
  Cpu,
  Clock,
  ShieldCheck,
  FileCheck
} from 'lucide-react';
import { HealthResponse, HardwareStatus } from '../types';

interface Props {
  health?: HealthResponse;
  onNavigate: (tab: string, file?: File) => void;
  onLoadSample: () => void;
}

export const Dashboard: React.FC<Props> = ({ health, onNavigate, onLoadSample }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        onNavigate('video', file);
      } else {
        onNavigate('image', file);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type.startsWith('video/')) {
        onNavigate('video', file);
      } else {
        onNavigate('image', file);
      }
    }
  };

  const hw: HardwareStatus | undefined = health?.hardware;
  const stats = health?.stats;

  return (
    <div className="flex flex-col gap-8 w-full px-4 sm:px-8 lg:px-12 py-8">
      {/* High-Contrast Hero Banner */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 glass-panel rounded-3xl p-8 border-2 border-rose-300 shadow-xl relative overflow-hidden bg-gradient-to-r from-pink-100 via-rose-50 to-pink-100 text-slate-900 w-full">
        <div className="flex flex-col gap-3.5 max-w-3xl z-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white border border-rose-300 text-rose-700 text-xs font-black shadow-xs w-fit">
            <Zap className="w-4 h-4 text-rose-600 fill-rose-600" /> CUDA C++ Pixel Engine
          </div>
          <h1 className="text-4xl md:text-5xl font-black tracking-tight text-slate-900 leading-tight">
            GPU-Accelerated Low-Light <br />
            <span className="text-rose-600">
              Photo & Video Enhancement
            </span>
          </h1>
          <p className="text-slate-700 text-sm leading-relaxed font-bold">
            Enhance underexposed dark photos and videos with high-throughput custom CUDA C++ kernels.
            Reduces noise, restores details, and optimizes contrast in real-time.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 z-10 w-full md:w-auto">
          <button
            onClick={() => onNavigate('image')}
            className="flex items-center justify-center gap-2.5 px-6 py-4 rounded-2xl bg-rose-600 hover:bg-rose-700 text-white font-black text-sm shadow-xl shadow-rose-200 transition-all hover:scale-105"
          >
            <ImageIcon className="w-5 h-5 text-white" /> Enhance Image
          </button>
          <button
            onClick={() => onNavigate('video')}
            className="flex items-center justify-center gap-2.5 px-6 py-4 rounded-2xl bg-slate-900 hover:bg-slate-800 text-white font-black text-sm shadow-xl transition-all hover:scale-105"
          >
            <Film className="w-5 h-5 text-white" /> Enhance Video
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="glass-card rounded-3xl p-6 border-2 border-rose-200 flex flex-col gap-2 bg-white shadow-sm">
          <div className="flex items-center justify-between text-slate-700 text-xs font-black uppercase tracking-wider">
            <span>ACTIVE BACKEND</span>
            <Cpu className="w-5 h-5 text-rose-600" />
          </div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-xl font-black text-slate-900 truncate">
              {hw?.gpu_available ? hw.device_name : 'CPU Fallback Engine'}
            </span>
          </div>
          <span className="text-xs text-rose-700 font-black">
            {hw?.gpu_available ? `CUDA Active | ${hw.vram_free_mb} MB VRAM` : 'Host CPU Active'}
          </span>
        </div>

        <div className="glass-card rounded-3xl p-6 border-2 border-rose-200 flex flex-col gap-2 bg-white shadow-sm">
          <div className="flex items-center justify-between text-slate-700 text-xs font-black uppercase tracking-wider">
            <span>FILES PROCESSED</span>
            <FileCheck className="w-5 h-5 text-rose-600" />
          </div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-mono font-black text-slate-900">
              {stats?.total_files_processed || 0}
            </span>
            <span className="text-xs font-bold text-slate-600">files</span>
          </div>
          <span className="text-xs text-slate-600 font-bold">Session counter</span>
        </div>

        <div className="glass-card rounded-3xl p-6 border-2 border-rose-200 flex flex-col gap-2 bg-white shadow-sm">
          <div className="flex items-center justify-between text-slate-700 text-xs font-black uppercase tracking-wider">
            <span>AVERAGE EXECUTION</span>
            <Clock className="w-5 h-5 text-rose-600" />
          </div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-mono font-black text-slate-900">
              {stats?.avg_processing_time_ms ? stats.avg_processing_time_ms.toFixed(2) : '0.00'}
            </span>
            <span className="text-xs font-bold text-slate-600">ms</span>
          </div>
          <span className="text-xs text-slate-600 font-bold">Per file execution</span>
        </div>

        <div className="glass-card rounded-3xl p-6 border-2 border-rose-200 flex flex-col gap-2 bg-white shadow-sm">
          <div className="flex items-center justify-between text-slate-700 text-xs font-black uppercase tracking-wider">
            <span>CUDA KERNELS STATUS</span>
            <ShieldCheck className="w-5 h-5 text-rose-600" />
          </div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-black text-rose-600">
              {hw?.compiled_kernels?.length || 0} Kernels
            </span>
          </div>
          <span className="text-xs text-slate-600 font-bold">Compiled & Loaded</span>
        </div>
      </div>

      {/* Main Drag and Drop Upload Area */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
        className="glass-card rounded-3xl p-12 border-3 border-dashed border-rose-300 hover:border-rose-500 transition-all flex flex-col items-center justify-center text-center gap-5 group cursor-pointer bg-gradient-to-b from-rose-50/50 to-white shadow-lg"
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="w-20 h-20 rounded-2xl bg-rose-600 border-2 border-white flex items-center justify-center group-hover:scale-110 transition-transform shadow-xl">
          <Upload className="w-9 h-9 text-white" />
        </div>

        <div className="flex flex-col gap-1 max-w-md">
          <h3 className="text-2xl font-black text-slate-900">
            Drag & Drop dark photo or video here
          </h3>
          <p className="text-xs text-slate-600 font-bold">
            Supports PNG, JPG, JPEG, BMP, MP4, AVI, MOV up to 4K resolution
          </p>
        </div>

        <div className="flex items-center gap-3 mt-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onLoadSample();
            }}
            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-black shadow-lg transition-all hover:scale-105"
          >
            <Sparkles className="w-4 h-4 text-white" />
            Try Sample Low-Light Photo
          </button>
        </div>
      </div>

      {/* Recent Processing Jobs Table */}
      <div className="glass-card rounded-3xl p-6 border-2 border-rose-200 flex flex-col gap-4 bg-white shadow-md">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-black text-slate-900 flex items-center gap-2">
            <Clock className="w-5 h-5 text-rose-600" />
            Recent Processing Jobs
          </h3>
          <span className="text-xs text-slate-600 font-mono font-bold">Session execution history</span>
        </div>

        {stats?.recent_jobs && stats.recent_jobs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b-2 border-rose-200 text-xs font-black text-slate-800 uppercase tracking-wider bg-rose-50">
                  <th className="py-4 px-4">Time</th>
                  <th className="py-4 px-4">Type</th>
                  <th className="py-4 px-4">File ID</th>
                  <th className="py-4 px-4">Dimensions</th>
                  <th className="py-4 px-4">Duration</th>
                  <th className="py-4 px-4">Backend</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rose-100 text-xs font-mono font-bold">
                {stats.recent_jobs.map((job) => (
                  <tr key={job.id} className="hover:bg-rose-50/60 transition-colors">
                    <td className="py-4 px-4 text-slate-600">{job.timestamp}</td>
                    <td className="py-4 px-4 font-sans font-black text-slate-900">{job.type}</td>
                    <td className="py-4 px-4 text-slate-900">{job.name}</td>
                    <td className="py-4 px-4 text-slate-600">{job.dimensions}</td>
                    <td className="py-4 px-4 text-rose-600 font-black">{job.duration_ms} ms</td>
                    <td className="py-4 px-4">
                      <span className="px-3 py-1 rounded-full bg-rose-600 text-white text-[10px] font-sans font-black uppercase tracking-wider shadow">
                        {job.backend}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center text-slate-500 text-xs font-bold">
            No processing jobs executed yet in this session.
          </div>
        )}
      </div>
    </div>
  );
};
