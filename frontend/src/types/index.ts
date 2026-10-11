export type ProcessingBackend = 'auto' | 'cuda' | 'cpu';

export interface HardwareStatus {
  gpu_available: boolean;
  device_name: string;
  vram_total_mb: number;
  vram_free_mb: number;
  backend: string;
  compiled_kernels: string[];
}

export interface SessionStats {
  total_files_processed: number;
  avg_processing_time_ms: number;
  recent_jobs: JobRecord[];
}

export interface JobRecord {
  id: string;
  timestamp: string;
  type: string;
  name: string;
  duration_ms: number;
  backend: string;
  dimensions: string;
}

export interface HealthResponse {
  status: string;
  hardware: HardwareStatus;
  stats: SessionStats;
}

export interface ImageUploadResponse {
  file_id: string;
  filename: string;
  width: number;
  height: number;
  channels: number;
  preview_url: string;
}

export interface ProcessingMetrics {
  kernel_time_ms: number;
  htod_ms: number;
  dtoh_ms: number;
  total_time_ms: number;
  backend: string;
  active_backend: string;
  gamma_used?: number;
  contrast_used?: number;
}

export interface ImageEnhanceResponse {
  file_id: string;
  original_url: string;
  enhanced_url: string;
  width: number;
  height: number;
  metrics: ProcessingMetrics;
}

export interface VideoUploadResponse {
  file_id: string;
  filename: string;
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  duration_sec: number;
  preview_url: string;
}

export interface VideoEnhanceResponse {
  job_id: string;
  output_file_id: string;
  status: string;
}

export interface VideoStatusResponse {
  status: 'processing' | 'completed' | 'failed' | 'not_found';
  progress_percentage: number;
  processed_frames: number;
  total_frames: number;
  fps: number;
  processing_fps: number;
  duration_sec: number;
  backend: string;
  result_url?: string;
  error?: string;
}

export interface BatchItemResult {
  filename: string;
  status: 'success' | 'failed';
  output_url?: string;
  metrics?: ProcessingMetrics;
  dimensions?: string;
  error?: string;
}

export interface BatchProcessResponse {
  results: BatchItemResult[];
  zip_url: string;
  total_files: number;
}

export interface OperationBenchmark {
  operation: string;
  cpu_time_ms: number;
  gpu_kernel_ms: number;
  gpu_htod_ms: number;
  gpu_dtoh_ms: number;
  gpu_total_ms: number;
  kernel_speedup: number;
  e2e_speedup: number;
}

export interface ResolutionBenchmark {
  resolution: string;
  dimensions: string;
  pixel_count: number;
  operations: OperationBenchmark[];
}

export interface BenchmarkResponse {
  hardware: HardwareStatus;
  benchmark_results: ResolutionBenchmark[];
  summary_note: string;
}
