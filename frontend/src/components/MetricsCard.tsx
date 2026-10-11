import React from 'react';
import { Clock, Cpu, HardDriveDownload, HardDriveUpload, Zap } from 'lucide-react';
import { ProcessingMetrics } from '../types';

interface Props {
  metrics?: ProcessingMetrics;
}

export const MetricsCard: React.FC<Props> = ({ metrics }) => {
  if (!metrics) return null;

  const isCuda = metrics.active_backend?.includes('CUDA') || metrics.backend?.includes('CUDA');

  return (
    <div className="glass-card rounded-3xl p-6 border-4 border-pink-300 flex flex-col gap-4 shadow-xl bg-white">
      <div className="flex items-center justify-between pb-3 border-b-2 border-pink-100">
        <div className="flex items-center gap-2">
          <Clock className="w-5 h-5 text-pink-600" />
          <h3 className="text-base font-black text-pink-950">Execution Diagnostics</h3>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-pink-100 text-xs font-mono font-black border border-pink-300">
          <span className="text-pink-700">Backend:</span>
          <span className={isCuda ? 'text-pink-600 font-extrabold' : 'text-amber-600 font-extrabold'}>
            {metrics.active_backend || metrics.backend}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Total Time */}
        <div className="bg-gradient-to-br from-pink-50 to-pink-100/60 p-4 rounded-2xl border-2 border-pink-200 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center gap-1.5 text-xs font-bold text-pink-700">
            <Zap className="w-4 h-4 text-pink-600 fill-pink-600" />
            Total End-to-End Latency
          </div>
          <span className="text-2xl font-mono font-black text-pink-950">
            {metrics.total_time_ms.toFixed(2)} <span className="text-xs font-bold text-pink-600">ms</span>
          </span>
        </div>

        {/* GPU Kernel Execution Time */}
        <div className="bg-gradient-to-br from-pink-50 to-pink-100/60 p-4 rounded-2xl border-2 border-pink-200 flex flex-col gap-1 shadow-sm">
          <div className="flex items-center gap-1.5 text-xs font-bold text-pink-700">
            <Cpu className="w-4 h-4 text-pink-600" />
            CUDA Kernel Pass Time
          </div>
          <span className="text-2xl font-mono font-black text-pink-600">
            {metrics.kernel_time_ms.toFixed(2)} <span className="text-xs font-bold text-pink-600">ms</span>
          </span>
        </div>
      </div>

      {/* Memory Transfer Breakdown (if CUDA) */}
      {isCuda && (
        <div className="grid grid-cols-2 gap-3 pt-2 border-t-2 border-pink-100">
          <div className="flex items-center justify-between text-xs px-3 py-2 rounded-xl bg-pink-50 text-pink-900 border border-pink-200 font-bold">
            <span className="flex items-center gap-1 text-pink-700">
              <HardDriveUpload className="w-3.5 h-3.5 text-pink-600" /> HtoD Copy:
            </span>
            <span className="font-mono font-black text-pink-950">{metrics.htod_ms.toFixed(2)} ms</span>
          </div>

          <div className="flex items-center justify-between text-xs px-3 py-2 rounded-xl bg-pink-50 text-pink-900 border border-pink-200 font-bold">
            <span className="flex items-center gap-1 text-pink-700">
              <HardDriveDownload className="w-3.5 h-3.5 text-pink-600" /> DtoH Copy:
            </span>
            <span className="font-mono font-black text-pink-950">{metrics.dtoh_ms.toFixed(2)} ms</span>
          </div>
        </div>
      )}
    </div>
  );
};
