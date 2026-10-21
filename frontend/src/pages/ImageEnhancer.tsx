import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Sliders,
  RotateCcw,
  Download,
  Sparkles,
  Zap,
  Cpu,
  Check,
  RefreshCw
} from 'lucide-react';
import { ImageUploadResponse, ImageEnhanceResponse, ProcessingBackend, HardwareStatus } from '../types';
import { uploadImage, enhanceImageParams, fetchSampleImage } from '../api/client';
import { ImageComparison } from '../components/ImageComparison';
import { MetricsCard } from '../components/MetricsCard';

interface Props {
  initialFile?: File;
  hardware?: HardwareStatus;
}

export const ImageEnhancer: React.FC<Props> = ({ initialFile, hardware }) => {
  const [currentImage, setCurrentImage] = useState<ImageUploadResponse | null>(null);
  const [enhancedResult, setEnhancedResult] = useState<ImageEnhanceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Enhancement Sliders State
  const [gamma, setGamma] = useState<number>(2.0);
  const [brightness, setBrightness] = useState<number>(10.0);
  const [contrast, setContrast] = useState<number>(1.25);
  const [denoising, setDenoising] = useState<number>(0.5);
  const [sharpening, setSharpening] = useState<number>(0.5);
  const [useClahe, setUseClahe] = useState<boolean>(false);
  const [autoEnhance, setAutoEnhance] = useState<boolean>(false);
  const [backend, setBackend] = useState<ProcessingBackend>('auto');

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (initialFile) {
      handleUpload(initialFile);
    } else {
      loadSample();
    }
  }, [initialFile]);

  const loadSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const sample = await fetchSampleImage();
      setCurrentImage(sample);
      processEnhancement(sample.file_id, gamma, brightness, contrast, denoising, sharpening, useClahe, autoEnhance, backend);
    } catch (err: any) {
      setError(err.message || 'Failed to load sample image');
      setLoading(false);
    }
  };

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const uploadRes = await uploadImage(file);
      setCurrentImage(uploadRes);
      processEnhancement(uploadRes.file_id, gamma, brightness, contrast, denoising, sharpening, useClahe, autoEnhance, backend);
    } catch (err: any) {
      setError(err.message || 'Image upload failed');
      setLoading(false);
    }
  };

  const processEnhancement = async (
    file_id: string,
    g: number,
    b: number,
    c: number,
    d: number,
    s: number,
    clahe: boolean,
    auto: boolean,
    backend_pref: ProcessingBackend
  ) => {
    setLoading(true);
    setError(null);
    try {
      const res = await enhanceImageParams({
        file_id,
        gamma: g,
        brightness: b,
        contrast: c,
        denoising_strength: d,
        sharpening_strength: s,
        use_clahe: clahe,
        auto_enhance: auto,
        backend_preference: backend_pref,
      });
      setEnhancedResult(res);
    } catch (err: any) {
      setError(err.message || 'Enhancement processing failed');
    } finally {
      setLoading(false);
    }
  };

  const handleApply = () => {
    if (currentImage) {
      processEnhancement(currentImage.file_id, gamma, brightness, contrast, denoising, sharpening, useClahe, autoEnhance, backend);
    }
  };

  const handleReset = () => {
    setGamma(1.0);
    setBrightness(0.0);
    setContrast(1.0);
    setDenoising(0.0);
    setSharpening(0.0);
    setUseClahe(false);
    setAutoEnhance(false);
    if (currentImage) {
      processEnhancement(currentImage.file_id, 1.0, 0.0, 1.0, 0.0, 0.0, false, false, backend);
    }
  };

  const handleAutoEnhanceToggle = () => {
    const nextAuto = !autoEnhance;
    setAutoEnhance(nextAuto);
    if (nextAuto) {
      setGamma(2.2);
      setContrast(1.35);
      setBrightness(15.0);
      setDenoising(0.5);
      setSharpening(0.5);
      if (currentImage) {
        processEnhancement(currentImage.file_id, 2.2, 15.0, 1.35, 0.5, 0.5, useClahe, true, backend);
      }
    }
  };

  return (
    <div className="w-full px-4 sm:px-8 lg:px-12 py-6 flex flex-col gap-6">
      {/* Top Action Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 glass-panel rounded-2xl p-4 border-2 border-rose-200 bg-white">
        <div className="flex items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-black text-xs shadow-md shadow-rose-200 transition-all"
          >
            <Upload className="w-4 h-4" /> Upload New Image
          </button>
          <button
            onClick={loadSample}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-bold text-xs shadow-md transition-all"
          >
            <Sparkles className="w-4 h-4 text-rose-400" /> Sample Low-Light Photo
          </button>
        </div>

        {enhancedResult && (
          <a
            href={enhancedResult.enhanced_url}
            download={`nightvision_${enhancedResult.file_id}`}
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-black text-xs shadow-lg shadow-rose-200 transition-all hover:scale-105"
          >
            <Download className="w-4 h-4" /> Download Enhanced Image
          </a>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border-2 border-rose-300 text-rose-800 text-xs font-black">
          Error: {error}
        </div>
      )}

      {/* Main Studio Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Sidebar: Controls */}
        <div className="lg:col-span-4 flex flex-col gap-5 glass-card rounded-2xl p-5 border-2 border-rose-200 shadow-md bg-white">
          <div className="flex items-center justify-between pb-3 border-b-2 border-rose-100">
            <div className="flex items-center gap-2">
              <Sliders className="w-5 h-5 text-rose-600" />
              <h2 className="text-base font-black text-slate-900">Enhancement Controls</h2>
            </div>
            <button
              onClick={handleReset}
              className="flex items-center gap-1 text-xs text-rose-600 font-bold hover:text-rose-800 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          </div>

          {/* Auto Enhance Preset Toggle */}
          <button
            onClick={handleAutoEnhanceToggle}
            className={`w-full flex items-center justify-between px-4 py-3 rounded-xl border-2 text-xs font-black transition-all ${
              autoEnhance
                ? 'bg-rose-600 border-rose-700 text-white shadow-md'
                : 'bg-rose-50/60 border-rose-200 text-slate-900 hover:bg-rose-100'
            }`}
          >
            <span className="flex items-center gap-2">
              <Sparkles className={`w-4 h-4 ${autoEnhance ? 'text-white' : 'text-rose-600'}`} />
              Auto Enhance Preset
            </span>
            {autoEnhance && <Check className="w-4 h-4 text-white" />}
          </button>

          {/* Processing Backend Selector */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-black text-slate-800 flex items-center gap-1.5 uppercase tracking-wider">
              <Cpu className="w-4 h-4 text-rose-600" /> Processing Backend
            </label>
            <div className="grid grid-cols-3 gap-2 bg-rose-50 p-1.5 rounded-xl border border-rose-200">
              <button
                onClick={() => setBackend('auto')}
                className={`py-2 rounded-lg text-xs font-black transition-all ${
                  backend === 'auto' ? 'bg-rose-600 text-white shadow' : 'text-slate-800 hover:text-rose-600'
                }`}
              >
                Auto
              </button>
              <button
                onClick={() => setBackend('cuda')}
                disabled={!hardware?.gpu_available}
                className={`py-2 rounded-lg text-xs font-black transition-all ${
                  backend === 'cuda'
                    ? 'bg-rose-600 text-white shadow'
                    : hardware?.gpu_available
                    ? 'text-slate-800 hover:text-rose-600'
                    : 'text-slate-400 cursor-not-allowed'
                }`}
              >
                CUDA GPU
              </button>
              <button
                onClick={() => setBackend('cpu')}
                className={`py-2 rounded-lg text-xs font-black transition-all ${
                  backend === 'cpu' ? 'bg-rose-600 text-white shadow' : 'text-slate-800 hover:text-rose-600'
                }`}
              >
                CPU
              </button>
            </div>
          </div>

          {/* Sliders */}
          <div className="flex flex-col gap-4">
            {/* Gamma Slider */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-extrabold text-slate-900">Gamma Correction</span>
                <span className="font-mono text-rose-600 font-black">{gamma.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="3.5"
                step="0.05"
                value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                className="w-full h-2 bg-rose-100 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Brightness Slider */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-extrabold text-slate-900">Brightness</span>
                <span className="font-mono text-rose-600 font-black">{brightness > 0 ? `+${brightness}` : brightness}</span>
              </div>
              <input
                type="range"
                min="-50"
                max="50"
                step="1"
                value={brightness}
                onChange={(e) => setBrightness(parseFloat(e.target.value))}
                className="w-full h-2 bg-rose-100 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Contrast Slider */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-extrabold text-slate-900">Contrast</span>
                <span className="font-mono text-rose-600 font-black">{contrast.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.5"
                step="0.05"
                value={contrast}
                onChange={(e) => setContrast(parseFloat(e.target.value))}
                className="w-full h-2 bg-rose-100 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Denoising Strength */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-extrabold text-slate-900">Gaussian Denoising</span>
                <span className="font-mono text-rose-600 font-black">{denoising.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={denoising}
                onChange={(e) => setDenoising(parseFloat(e.target.value))}
                className="w-full h-2 bg-rose-100 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* Sharpening Strength */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-extrabold text-slate-900">Sharpening Strength</span>
                <span className="font-mono text-rose-600 font-black">{sharpening.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="2.0"
                step="0.05"
                value={sharpening}
                onChange={(e) => setSharpening(parseFloat(e.target.value))}
                className="w-full h-2 bg-rose-100 rounded-lg appearance-none cursor-pointer accent-rose-600"
              />
            </div>

            {/* CLAHE Checkbox */}
            <label className="flex items-center gap-2 cursor-pointer pt-2">
              <input
                type="checkbox"
                checked={useClahe}
                onChange={(e) => setUseClahe(e.target.checked)}
                className="w-4 h-4 rounded bg-white border-rose-300 text-rose-600 focus:ring-0 cursor-pointer"
              />
              <span className="text-xs font-black text-slate-900">CLAHE Adaptive Histogram Equalization</span>
            </label>
          </div>

          <button
            onClick={handleApply}
            disabled={loading || !currentImage}
            className="w-full py-3.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-black text-xs shadow-lg shadow-rose-200 transition-all flex items-center justify-center gap-2 mt-2"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
                Executing CUDA Kernel...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-white fill-white" />
                Apply CUDA Enhancement
              </>
            )}
          </button>
        </div>

        {/* Center/Right: Image View Workspace & Diagnostics */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          {enhancedResult && currentImage ? (
            <>
              <ImageComparison
                originalUrl={enhancedResult.original_url}
                enhancedUrl={enhancedResult.enhanced_url}
                originalDimensions={`${currentImage.width}x${currentImage.height}`}
                enhancedDimensions={`${enhancedResult.width}x${enhancedResult.height}`}
              />
              <MetricsCard metrics={enhancedResult.metrics} />
            </>
          ) : (
            <div className="w-full h-[520px] glass-card rounded-2xl border-2 border-rose-200 flex flex-col items-center justify-center gap-4 bg-white">
              <RefreshCw className="w-8 h-8 text-rose-600 animate-spin" />
              <p className="text-xs text-slate-700 font-mono font-bold">Loading image workspace...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
