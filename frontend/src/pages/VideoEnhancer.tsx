import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Film,
  Play,
  Download,
  Zap,
  RefreshCw,
  Cpu,
  AlertCircle
} from 'lucide-react';
import { VideoUploadResponse, VideoStatusResponse, ProcessingBackend, HardwareStatus } from '../types';
import { uploadVideo, enhanceVideoJob, fetchVideoStatus } from '../api/client';

interface Props {
  initialFile?: File;
  hardware?: HardwareStatus;
}

export const VideoEnhancer: React.FC<Props> = ({ initialFile, hardware }) => {
  const [videoInfo, setVideoInfo] = useState<VideoUploadResponse | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<VideoStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Sliders
  const [gamma, setGamma] = useState<number>(2.0);
  const [brightness, setBrightness] = useState<number>(10.0);
  const [contrast, setContrast] = useState<number>(1.25);
  const [denoising, setDenoising] = useState<number>(0.5);
  const [sharpening, setSharpening] = useState<number>(0.5);
  const [backend, setBackend] = useState<ProcessingBackend>('auto');

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (initialFile) {
      handleUpload(initialFile);
    }
  }, [initialFile]);

  useEffect(() => {
    let interval: any;
    if (jobId && jobStatus?.status === 'processing') {
      interval = setInterval(async () => {
        try {
          const status = await fetchVideoStatus(jobId);
          setJobStatus(status);
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(interval);
          }
        } catch (err) {
          console.error('Error fetching video status:', err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [jobId, jobStatus?.status]);

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setJobId(null);
    setJobStatus(null);
    try {
      const res = await uploadVideo(file);
      setVideoInfo(res);
    } catch (err: any) {
      setError(err.message || 'Failed to upload video');
    } finally {
      setLoading(false);
    }
  };

  const handleStartProcessing = async () => {
    if (!videoInfo) return;
    setLoading(true);
    setError(null);
    try {
      const jobRes = await enhanceVideoJob({
        file_id: videoInfo.file_id,
        gamma,
        brightness,
        contrast,
        denoising_strength: denoising,
        sharpening_strength: sharpening,
        backend_preference: backend,
      });
      setJobId(jobRes.job_id);
      setJobStatus({
        status: 'processing',
        progress_percentage: 0,
        processed_frames: 0,
        total_frames: videoInfo.total_frames,
        fps: videoInfo.fps,
        processing_fps: 0,
        duration_sec: videoInfo.duration_sec,
        backend: hardware?.gpu_available && backend !== 'cpu' ? 'CUDA GPU' : 'CPU',
      });
    } catch (err: any) {
      setError(err.message || 'Failed to start video processing job');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full px-4 sm:px-8 lg:px-12 py-6 flex flex-col gap-6">
      {/* Upload Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 glass-panel rounded-2xl p-4 border border-pink-200">
        <div className="flex items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-white font-extrabold text-xs shadow-md shadow-pink-200 transition-all"
          >
            <Upload className="w-4 h-4 text-white" /> Select Input Video
          </button>
          {videoInfo && (
            <span className="text-xs font-mono text-slate-700">
              Selected: <strong className="text-rose-600">{videoInfo.filename}</strong> ({videoInfo.width}x{videoInfo.height} @ {videoInfo.fps} FPS, {videoInfo.duration_sec}s)
            </span>
          )}
        </div>

        {jobStatus?.status === 'completed' && jobStatus.result_url && (
          <a
            href={jobStatus.result_url}
            download={`enhanced_${videoInfo?.filename || 'video.mp4'}`}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-rose-500 to-pink-600 text-white font-extrabold text-xs shadow-lg shadow-pink-200 transition-all hover:scale-105"
          >
            <Download className="w-4 h-4 text-white" /> Download Enhanced MP4
          </a>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-500" />
          {error}
        </div>
      )}

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-5 glass-card rounded-2xl p-5 border border-pink-200 shadow-sm bg-white">
          <div className="flex items-center justify-between pb-3 border-b border-pink-100">
            <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
              <Film className="w-4 h-4 text-rose-500" /> Frame Processing Parameters
            </h2>
          </div>

          {/* Backend Selector */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-600 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-rose-500" /> Acceleration Mode
            </label>
            <div className="grid grid-cols-3 gap-2 bg-pink-50 p-1 rounded-xl border border-pink-200">
              <button
                onClick={() => setBackend('auto')}
                className={`py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  backend === 'auto' ? 'bg-rose-500 text-white font-bold shadow-sm' : 'text-slate-600 hover:text-rose-600'
                }`}
              >
                Auto
              </button>
              <button
                onClick={() => setBackend('cuda')}
                disabled={!hardware?.gpu_available}
                className={`py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  backend === 'cuda'
                    ? 'bg-rose-500 text-white font-bold shadow-sm'
                    : hardware?.gpu_available
                    ? 'text-slate-600 hover:text-rose-600'
                    : 'text-slate-400 cursor-not-allowed'
                }`}
              >
                CUDA GPU
              </button>
              <button
                onClick={() => setBackend('cpu')}
                className={`py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  backend === 'cpu' ? 'bg-rose-500 text-white font-bold shadow-sm' : 'text-slate-600 hover:text-rose-600'
                }`}
              >
                CPU
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            {/* Gamma */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-700">Gamma Correction</span>
                <span className="font-mono text-rose-600 font-bold">{gamma.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="3.5"
                step="0.05"
                value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-pink-100 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
            </div>

            {/* Brightness */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-700">Brightness</span>
                <span className="font-mono text-rose-600 font-bold">{brightness}</span>
              </div>
              <input
                type="range"
                min="-50"
                max="50"
                step="1"
                value={brightness}
                onChange={(e) => setBrightness(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-pink-100 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
            </div>

            {/* Contrast */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-700">Contrast</span>
                <span className="font-mono text-rose-600 font-bold">{contrast.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.5"
                step="0.05"
                value={contrast}
                onChange={(e) => setContrast(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-pink-100 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
            </div>

            {/* Denoising */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-700">Denoising</span>
                <span className="font-mono text-rose-600 font-bold">{denoising.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={denoising}
                onChange={(e) => setDenoising(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-pink-100 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
            </div>
          </div>

          <button
            onClick={handleStartProcessing}
            disabled={!videoInfo || loading || jobStatus?.status === 'processing'}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white font-extrabold text-xs shadow-lg shadow-pink-200 transition-all flex items-center justify-center gap-2 mt-2"
          >
            {jobStatus?.status === 'processing' ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
                Processing Video Frames...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 text-white fill-white" />
                Start CUDA Video Enhancement
              </>
            )}
          </button>
        </div>

        {/* Video Workspace & Progress Status */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          {/* Progress Indicator Card */}
          {jobStatus && (
            <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-md bg-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-rose-500" />
                  <h3 className="text-sm font-bold text-slate-800">Video Enhancement Progress</h3>
                </div>
                <span className="text-xs font-mono font-bold text-rose-600">
                  {jobStatus.progress_percentage.toFixed(1)}%
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-3 bg-pink-100 rounded-full overflow-hidden border border-pink-200 p-0.5">
                <div
                  className="h-full bg-gradient-to-r from-rose-500 to-pink-500 rounded-full transition-all duration-300 shadow-sm"
                  style={{ width: `${jobStatus.progress_percentage}%` }}
                ></div>
              </div>

              <div className="grid grid-cols-3 gap-4 pt-2 font-mono text-xs text-slate-700">
                <div className="flex flex-col">
                  <span className="text-[11px] text-slate-500 font-sans">Frames Processed</span>
                  <span className="font-bold text-slate-900">
                    {jobStatus.processed_frames} / {jobStatus.total_frames}
                  </span>
                </div>

                <div className="flex flex-col">
                  <span className="text-[11px] text-slate-500 font-sans">Processing Speed</span>
                  <span className="font-bold text-rose-600">{jobStatus.processing_fps} FPS</span>
                </div>

                <div className="flex flex-col">
                  <span className="text-[11px] text-slate-500 font-sans">Backend</span>
                  <span className="font-bold text-rose-600">{jobStatus.backend}</span>
                </div>
              </div>
            </div>
          )}

          {/* Video Preview Player */}
          {videoInfo ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original Video */}
              <div className="glass-card rounded-2xl p-4 border border-pink-200 flex flex-col gap-2 bg-slate-900">
                <span className="text-xs font-bold text-white">Original Dark Video</span>
                <video
                  src={videoInfo.preview_url}
                  controls
                  className="w-full h-64 rounded-xl bg-black object-contain"
                />
              </div>

              {/* Enhanced Video */}
              <div className="glass-card rounded-2xl p-4 border border-pink-300 flex flex-col gap-2 bg-slate-900">
                <span className="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-rose-400" /> Enhanced Video Output
                </span>
                {jobStatus?.status === 'completed' && jobStatus.result_url ? (
                  <video
                    src={jobStatus.result_url}
                    controls
                    autoPlay
                    className="w-full h-64 rounded-xl bg-black object-contain"
                  />
                ) : (
                  <div className="w-full h-64 rounded-xl bg-slate-950 border border-slate-800 flex flex-col items-center justify-center gap-3 text-slate-400 text-xs">
                    {jobStatus?.status === 'processing' ? (
                      <>
                        <RefreshCw className="w-6 h-6 text-rose-500 animate-spin" />
                        Rendering frames on GPU...
                      </>
                    ) : (
                      'Processed video player will appear here'
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="w-full h-[400px] glass-card rounded-2xl border border-pink-200 flex flex-col items-center justify-center gap-3 text-slate-400 bg-white">
              <Film className="w-10 h-10 text-pink-300" />
              <p className="text-xs font-mono text-slate-500">Upload an MP4 or AVI video to open video studio</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
