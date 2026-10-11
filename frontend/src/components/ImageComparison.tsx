import React, { useState, useRef } from 'react';
import { Columns, Split, Eye, Sparkles } from 'lucide-react';

interface Props {
  originalUrl: string;
  enhancedUrl: string;
  originalDimensions?: string;
  enhancedDimensions?: string;
}

export const ImageComparison: React.FC<Props> = ({
  originalUrl,
  enhancedUrl,
  originalDimensions = '1280x720',
  enhancedDimensions = '1280x720',
}) => {
  const [mode, setMode] = useState<'slider' | 'side-by-side'>('slider');
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const isDragging = useRef<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  return (
    <div className="flex flex-col gap-3 w-full h-full">
      {/* View Mode Toolbar */}
      <div className="flex items-center justify-between px-5 py-3 glass-card rounded-2xl border-2 border-pink-300 shadow-lg bg-white">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setMode('slider')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all ${
              mode === 'slider'
                ? 'bg-gradient-to-r from-pink-600 to-rose-600 text-white shadow-md'
                : 'text-pink-700 hover:bg-pink-100 font-bold'
            }`}
          >
            <Split className="w-4 h-4" />
            Interactive Split Slider
          </button>
          <button
            onClick={() => setMode('side-by-side')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all ${
              mode === 'side-by-side'
                ? 'bg-gradient-to-r from-pink-600 to-rose-600 text-white shadow-md'
                : 'text-pink-700 hover:bg-pink-100 font-bold'
            }`}
          >
            <Columns className="w-4 h-4" />
            Side by Side View
          </button>
        </div>

        <div className="flex items-center gap-5 text-xs font-mono font-black text-pink-900">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
            Original: <strong>{originalDimensions}</strong>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-pink-600"></span>
            Enhanced: <strong className="text-pink-600">{enhancedDimensions}</strong>
          </div>
        </div>
      </div>

      {/* Main Image Stage */}
      {mode === 'slider' ? (
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onMouseMove={handleMouseMove}
          onTouchMove={handleTouchMove}
          className="relative w-full h-[540px] rounded-3xl overflow-hidden glass-card select-none cursor-ew-resize border-4 border-pink-400 shadow-2xl bg-black"
        >
          {/* Enhanced Image (Background) */}
          <img
            src={enhancedUrl}
            alt="Enhanced"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />

          {/* Original Image (Clipped Foreground) */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ width: `${sliderPosition}%` }}
          >
            <img
              src={originalUrl}
              alt="Original"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none max-w-none"
              style={{ width: containerRef.current?.clientWidth || '100%' }}
            />
          </div>

          {/* Slider Line & Handle */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-gradient-to-b from-white via-pink-400 to-pink-600 shadow-xl pointer-events-none"
            style={{ left: `${sliderPosition}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-10 h-10 rounded-full bg-white border-4 border-pink-600 flex items-center justify-center shadow-2xl">
              <Split className="w-5 h-5 text-pink-600" />
            </div>
          </div>

          {/* Labels */}
          <div className="absolute top-5 left-5 px-4 py-2 rounded-full bg-black/80 text-white text-xs font-black border border-white/40 shadow-lg">
            <Eye className="w-4 h-4 inline mr-1.5 text-pink-300" /> Original Dark Input
          </div>
          <div className="absolute top-5 right-5 px-4 py-2 rounded-full bg-gradient-to-r from-pink-600 to-rose-600 text-white text-xs font-black border-2 border-white shadow-xl">
            <Sparkles className="w-4 h-4 inline mr-1.5 text-white" /> CUDA Enhanced
          </div>
        </div>
      ) : (
        /* Side-by-Side Mode */
        <div className="grid grid-cols-2 gap-4 w-full h-[540px]">
          <div className="relative rounded-3xl overflow-hidden glass-card border-4 border-slate-700 flex flex-col bg-black">
            <div className="absolute top-4 left-4 z-10 px-4 py-2 rounded-full bg-black/80 text-white text-xs font-black border border-white/40">
              Original Input
            </div>
            <img
              src={originalUrl}
              alt="Original"
              className="w-full h-full object-contain p-3"
            />
          </div>

          <div className="relative rounded-3xl overflow-hidden glass-card border-4 border-pink-500 flex flex-col bg-black">
            <div className="absolute top-4 left-4 z-10 px-4 py-2 rounded-full bg-gradient-to-r from-pink-600 to-rose-600 text-white text-xs font-black border border-white">
              Enhanced Output
            </div>
            <img
              src={enhancedUrl}
              alt="Enhanced"
              className="w-full h-full object-contain p-3"
            />
          </div>
        </div>
      )}
    </div>
  );
};
