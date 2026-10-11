import React from 'react';
import { Cpu, Zap, AlertTriangle } from 'lucide-react';
import { HardwareStatus } from '../types';

interface Props {
  hardware?: HardwareStatus;
}

export const BackendStatusBadge: React.FC<Props> = ({ hardware }) => {
  if (!hardware) {
    return (
      <div className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-white text-pink-700 text-xs font-black shadow-md animate-pulse">
        <Cpu className="w-3.5 h-3.5 text-pink-600" />
        Detecting GPU...
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {hardware.gpu_available ? (
        <div className="flex items-center gap-2.5 px-4 py-2 rounded-full bg-white text-pink-900 text-xs font-black shadow-lg border-2 border-pink-300">
          <Zap className="w-4 h-4 text-pink-600 fill-pink-600 animate-bounce" />
          <span>{hardware.device_name}</span>
          <span className="text-pink-300">|</span>
          <span className="text-pink-600 font-mono font-black">{hardware.vram_free_mb} MB Free</span>
          <span className="px-2.5 py-0.5 rounded-full bg-gradient-to-r from-pink-600 to-rose-600 text-white text-[10px] font-black uppercase tracking-wider shadow">
            CUDA Active
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-amber-100 text-amber-900 text-xs font-black shadow">
          <AlertTriangle className="w-4 h-4 text-amber-600" />
          <span>CPU Mode (GPU Unavailable)</span>
        </div>
      )}
    </div>
  );
};
