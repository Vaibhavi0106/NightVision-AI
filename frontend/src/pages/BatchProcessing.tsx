import React, { useState, useRef } from 'react';
import {
  Upload,
  Layers,
  Archive,
  CheckCircle2,
  AlertCircle,
  Download,
  RefreshCw,
  Zap,
  FileCheck
} from 'lucide-react';
import { BatchProcessResponse, ProcessingBackend, HardwareStatus } from '../types';
import { processBatchImages } from '../api/client';

interface Props {
  hardware?: HardwareStatus;
}

export const BatchProcessing: React.FC<Props> = ({ hardware }) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [batchResponse, setBatchResponse] = useState<BatchProcessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Enhancement parameters
  const [gamma, setGamma] = useState<number>(2.0);
  const [brightness, setBrightness] = useState<number>(10.0);
  const [contrast, setContrast] = useState<number>(1.25);
  const [denoising, setDenoising] = useState<number>(0.5);
  const [sharpening, setSharpening] = useState<number>(0.5);
  const [backend, setBackend] = useState<ProcessingBackend>('auto');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSelectFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      setSelectedFiles(filesArray);
      setBatchResponse(null);
    }
  };

  const handleStartBatch = async () => {
    if (selectedFiles.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const res = await processBatchImages({
        files: selectedFiles,
        gamma,
        brightness,
        contrast,
        denoising_strength: denoising,
        sharpening_strength: sharpening,
        backend_preference: backend,
      });
      setBatchResponse(res);
    } catch (err: any) {
      setError(err.message || 'Batch image processing failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full px-4 sm:px-8 lg:px-12 py-6 flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 glass-panel rounded-2xl p-4 border border-pink-200">
        <div className="flex items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*"
            onChange={handleSelectFiles}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-white font-extrabold text-xs shadow-md shadow-pink-200 transition-all"
          >
            <Upload className="w-4 h-4 text-white" /> Select Multiple Photos
          </button>
          <span className="text-xs text-slate-700 font-mono">
            {selectedFiles.length > 0 ? `${selectedFiles.length} files selected for batch queue` : 'No files selected'}
          </span>
        </div>

        {batchResponse && (
          <a
            href={batchResponse.zip_url}
            download
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white font-extrabold text-xs shadow-lg shadow-pink-200 transition-all hover:scale-105"
          >
            <Archive className="w-4 h-4 text-white" /> Download All as ZIP Archive
          </a>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold">
          Error: {error}
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-5 glass-card rounded-2xl p-5 border border-pink-200 shadow-sm bg-white">
          <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2 pb-3 border-b border-pink-100">
            <Layers className="w-4 h-4 text-rose-500" /> Batch Enhancement Preset
          </h2>

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
            onClick={handleStartBatch}
            disabled={selectedFiles.length === 0 || loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white font-extrabold text-xs shadow-lg shadow-pink-200 transition-all flex items-center justify-center gap-2 mt-2"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
                Batch Processing Queue...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-white fill-white" />
                Process {selectedFiles.length} Photos on GPU
              </>
            )}
          </button>
        </div>

        {/* Queue Table */}
        <div className="lg:col-span-8 glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-rose-500" /> Batch Execution Queue
            </h3>
            <span className="text-xs text-slate-500 font-mono">
              {batchResponse ? `${batchResponse.results.length} processed` : `${selectedFiles.length} queued`}
            </span>
          </div>

          {selectedFiles.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-pink-100 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    <th className="py-3 px-4">Filename</th>
                    <th className="py-3 px-4">Size</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4">Duration</th>
                    <th className="py-3 px-4">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-pink-100 text-xs font-mono">
                  {selectedFiles.map((file, idx) => {
                    const result = batchResponse?.results.find((r) => r.filename === file.name);
                    return (
                      <tr key={idx} className="hover:bg-pink-50/50 transition-colors">
                        <td className="py-3 px-4 font-sans font-semibold text-slate-800">{file.name}</td>
                        <td className="py-3 px-4 text-slate-500 font-mono">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </td>
                        <td className="py-3 px-4">
                          {result ? (
                            result.status === 'success' ? (
                              <span className="inline-flex items-center gap-1 text-emerald-600 font-semibold font-sans">
                                <CheckCircle2 className="w-3.5 h-3.5" /> Completed
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-rose-600 font-semibold font-sans">
                                <AlertCircle className="w-3.5 h-3.5" /> Failed
                              </span>
                            )
                          ) : loading ? (
                            <span className="inline-flex items-center gap-1 text-rose-500 font-sans animate-pulse">
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" /> In Queue...
                            </span>
                          ) : (
                            <span className="text-slate-400 font-sans">Ready</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-rose-600 font-bold">
                          {result?.metrics ? `${result.metrics.total_time_ms.toFixed(2)} ms` : '-'}
                        </td>
                        <td className="py-3 px-4">
                          {result?.output_url ? (
                            <a
                              href={result.output_url}
                              download
                              className="inline-flex items-center gap-1 text-rose-600 hover:text-rose-700 font-sans font-semibold text-xs"
                            >
                              <Download className="w-3.5 h-3.5" /> Download
                            </a>
                          ) : (
                            '-'
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-12 text-center text-slate-400 text-xs">
              Select multiple photo files to begin batch enhancement
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
