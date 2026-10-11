import {
  HealthResponse,
  ImageUploadResponse,
  ImageEnhanceResponse,
  VideoUploadResponse,
  VideoEnhanceResponse,
  VideoStatusResponse,
  BatchProcessResponse,
  BenchmarkResponse,
  ProcessingBackend
} from '../types';

const API_BASE = '';

export async function fetchHealthStatus(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Failed to connect to backend service');
  return res.json();
}

export async function uploadImage(file: File): Promise<ImageUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/image/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Image upload failed');
  }
  return res.json();
}

export async function fetchSampleImage(): Promise<ImageUploadResponse> {
  const res = await fetch(`${API_BASE}/api/image/sample`);
  if (!res.ok) throw new Error('Failed to load sample image');
  return res.json();
}

export async function enhanceImageParams(params: {
  file_id: string;
  gamma: number;
  brightness: number;
  contrast: number;
  denoising_strength: number;
  sharpening_strength: number;
  use_clahe: boolean;
  auto_enhance: boolean;
  backend_preference: ProcessingBackend;
}): Promise<ImageEnhanceResponse> {
  const formData = new FormData();
  formData.append('file_id', params.file_id);
  formData.append('gamma', params.gamma.toString());
  formData.append('brightness', params.brightness.toString());
  formData.append('contrast', params.contrast.toString());
  formData.append('denoising_strength', params.denoising_strength.toString());
  formData.append('sharpening_strength', params.sharpening_strength.toString());
  formData.append('use_clahe', params.use_clahe ? 'true' : 'false');
  formData.append('auto_enhance', params.auto_enhance ? 'true' : 'false');
  formData.append('backend_preference', params.backend_preference);

  const res = await fetch(`${API_BASE}/api/image/enhance`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Enhancement failed');
  }
  return res.json();
}

export async function uploadVideo(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/video/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Video upload failed');
  }
  return res.json();
}

export async function enhanceVideoJob(params: {
  file_id: string;
  gamma: number;
  brightness: number;
  contrast: number;
  denoising_strength: number;
  sharpening_strength: number;
  backend_preference: ProcessingBackend;
}): Promise<VideoEnhanceResponse> {
  const formData = new FormData();
  formData.append('file_id', params.file_id);
  formData.append('gamma', params.gamma.toString());
  formData.append('brightness', params.brightness.toString());
  formData.append('contrast', params.contrast.toString());
  formData.append('denoising_strength', params.denoising_strength.toString());
  formData.append('sharpening_strength', params.sharpening_strength.toString());
  formData.append('backend_preference', params.backend_preference);

  const res = await fetch(`${API_BASE}/api/video/enhance`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to start video processing');
  }
  return res.json();
}

export async function fetchVideoStatus(job_id: string): Promise<VideoStatusResponse> {
  const res = await fetch(`${API_BASE}/api/video/status/${job_id}`);
  if (!res.ok) throw new Error('Failed to fetch video job status');
  return res.json();
}

export async function processBatchImages(params: {
  files: File[];
  gamma: number;
  brightness: number;
  contrast: number;
  denoising_strength: number;
  sharpening_strength: number;
  backend_preference: ProcessingBackend;
}): Promise<BatchProcessResponse> {
  const formData = new FormData();
  params.files.forEach((f) => formData.append('files', f));
  formData.append('gamma', params.gamma.toString());
  formData.append('brightness', params.brightness.toString());
  formData.append('contrast', params.contrast.toString());
  formData.append('denoising_strength', params.denoising_strength.toString());
  formData.append('sharpening_strength', params.sharpening_strength.toString());
  formData.append('backend_preference', params.backend_preference);

  const res = await fetch(`${API_BASE}/api/batch/process`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Batch processing failed');
  }
  return res.json();
}

export async function runPerformanceBenchmark(file?: File): Promise<BenchmarkResponse> {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  }
  const res = await fetch(`${API_BASE}/api/benchmark/run`, {
    method: 'POST',
    body: file ? formData : undefined,
  });
  if (!res.ok) throw new Error('Performance benchmark failed');
  return res.json();
}
