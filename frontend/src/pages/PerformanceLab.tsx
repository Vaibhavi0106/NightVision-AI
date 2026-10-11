import React, { useState } from 'react';
import {
  Cpu,
  Zap,
  Play,
  RefreshCw,
  BarChart2,
  TrendingUp,
  Info
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { BenchmarkResponse, ResolutionBenchmark, HardwareStatus } from '../types';
import { runPerformanceBenchmark } from '../api/client';

interface Props {
  hardware?: HardwareStatus;
}

export const PerformanceLab: React.FC<Props> = ({ hardware }) => {
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedResolution, setSelectedResolution] = useState<string>('1080p');

  const handleRunBenchmark = async () => {
    setLoading(true);
    try {
      const data = await runPerformanceBenchmark();
      setBenchmarkData(data);
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const currentResData: ResolutionBenchmark | undefined = benchmarkData?.benchmark_results.find(
    (r) => r.resolution === selectedResolution
  );

  const chartData = currentResData?.operations.map((op) => ({
    name: op.operation,
    CPU: op.cpu_time_ms,
    'GPU Kernel': op.gpu_kernel_ms,
    'GPU Total': op.gpu_total_ms,
    Speedup: op.kernel_speedup,
  }));

  return (
    <div className="max-w-7xl mx-auto px-6 py-6 flex flex-col gap-6">
      {/* Hero Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 glass-panel rounded-2xl p-6 border border-pink-200">
        <div className="flex flex-col gap-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-pink-100 border border-pink-200 text-rose-700 text-xs font-semibold w-fit">
            <Cpu className="w-3.5 h-3.5 text-rose-500" /> CUDA Benchmarking Suite
          </div>
          <h1 className="text-2xl font-bold text-slate-900">Performance Lab</h1>
          <p className="text-slate-600 text-xs max-w-xl">
            Empirically measure and compare execution times between host CPU and custom CUDA C++ kernels on the NVIDIA RTX 4060.
          </p>
        </div>

        <button
          onClick={handleRunBenchmark}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white font-extrabold text-xs shadow-lg shadow-pink-200 transition-all hover:scale-105"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-white" />
              Running CUDA Benchmark Suite...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-white fill-white" />
              Run Empirical GPU vs CPU Benchmarks
            </>
          )}
        </button>
      </div>

      {benchmarkData ? (
        <div className="flex flex-col gap-6">
          {/* Resolution Selector Tabs */}
          <div className="flex items-center gap-2 bg-pink-50/80 p-1.5 rounded-xl border border-pink-200 w-fit">
            {benchmarkData.benchmark_results.map((r) => (
              <button
                key={r.resolution}
                onClick={() => setSelectedResolution(r.resolution)}
                className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  selectedResolution === r.resolution
                    ? 'bg-rose-500 text-white font-bold shadow-sm'
                    : 'text-slate-600 hover:text-rose-600 hover:bg-white/60'
                }`}
              >
                {r.resolution} ({r.dimensions})
              </button>
            ))}
          </div>

          {/* Recharts Bar Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Latency Comparison */}
            <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
              <div className="flex items-center justify-between pb-3 border-b border-pink-100">
                <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-rose-500" /> Latency Comparison ({selectedResolution})
                </h3>
                <span className="text-xs font-mono text-slate-500">Lower is better (ms)</span>
              </div>

              <div className="w-full h-80 pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffe4e6" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} unit="ms" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#ffffff', borderColor: '#fecdd3', borderRadius: '8px' }}
                      itemStyle={{ color: '#0f172a', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                    <Bar dataKey="CPU" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="GPU Total" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="GPU Kernel" fill="#ec4899" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: CUDA Speedup Factor */}
            <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
              <div className="flex items-center justify-between pb-3 border-b border-pink-100">
                <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-rose-500" /> CUDA Kernel Speedup Factor
                </h3>
                <span className="text-xs font-mono text-rose-600 font-bold">Speedup = CPU time / GPU time</span>
              </div>

              <div className="w-full h-80 pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffe4e6" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} unit="x" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#ffffff', borderColor: '#fecdd3', borderRadius: '8px' }}
                      itemStyle={{ color: '#e11d48', fontSize: '12px' }}
                    />
                    <Bar dataKey="Speedup" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Breakdown Table */}
          <div className="glass-card rounded-2xl p-6 border border-pink-200 flex flex-col gap-4 shadow-sm bg-white">
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2 pb-3 border-b border-pink-100">
              <Zap className="w-4 h-4 text-rose-500" /> Detailed Empirical Execution Breakdown ({selectedResolution})
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-pink-100 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    <th className="py-3 px-4">Operation</th>
                    <th className="py-3 px-4">CPU Execution</th>
                    <th className="py-3 px-4">GPU Kernel</th>
                    <th className="py-3 px-4">HtoD Transfer</th>
                    <th className="py-3 px-4">DtoH Transfer</th>
                    <th className="py-3 px-4">GPU End-to-End</th>
                    <th className="py-3 px-4">Kernel Speedup</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-pink-100 text-xs font-mono">
                  {currentResData?.operations.map((op, idx) => (
                    <tr key={idx} className="hover:bg-pink-50/50 transition-colors">
                      <td className="py-3 px-4 font-sans font-semibold text-slate-800">{op.operation}</td>
                      <td className="py-3 px-4 text-amber-600 font-bold">{op.cpu_time_ms.toFixed(2)} ms</td>
                      <td className="py-3 px-4 text-rose-600 font-bold">{op.gpu_kernel_ms.toFixed(2)} ms</td>
                      <td className="py-3 px-4 text-slate-500">{op.gpu_htod_ms.toFixed(2)} ms</td>
                      <td className="py-3 px-4 text-slate-500">{op.gpu_dtoh_ms.toFixed(2)} ms</td>
                      <td className="py-3 px-4 text-pink-600 font-bold">{op.gpu_total_ms.toFixed(2)} ms</td>
                      <td className="py-3 px-4 text-rose-600 font-bold font-sans">
                        <span className="px-2 py-0.5 rounded bg-pink-100 border border-pink-200 text-rose-700">
                          {op.kernel_speedup.toFixed(1)}x Speedup
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Academic Analysis & Memory Overhead Note */}
          <div className="glass-panel rounded-2xl p-6 border border-pink-200 flex flex-col gap-3 text-xs leading-relaxed text-slate-700 bg-white">
            <h4 className="font-bold text-rose-600 flex items-center gap-2 text-sm">
              <Info className="w-4 h-4 text-rose-500" /> Academic Performance Analysis & Memory Transfer Costs
            </h4>
            <p>{benchmarkData.summary_note}</p>
          </div>
        </div>
      ) : (
        <div className="w-full h-96 glass-card rounded-2xl border border-pink-200 flex flex-col items-center justify-center gap-4 text-center p-8 bg-white">
          <BarChart2 className="w-12 h-12 text-rose-300" />
          <div className="flex flex-col gap-1">
            <h3 className="text-base font-bold text-slate-800">No Benchmark Data Collected Yet</h3>
            <p className="text-xs text-slate-500">
              Click "Run Empirical GPU vs CPU Benchmarks" above to execute timed benchmark passes across 720p, 1080p, 1440p, and 4K resolutions on your GPU.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
