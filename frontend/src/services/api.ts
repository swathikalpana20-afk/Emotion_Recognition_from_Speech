// EMOTISENSE AI — API service layer
import axios, { AxiosError } from 'axios';
import type {
  DatasetStatusResponse,
  HealthResponse,
  MetricsResponse,
  ModelStatusResponse,
  PredictionResponse,
  TrainRequest,
  TrainResponse,
  TrainStatusResponse,
} from '../types/api';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

// ── Error extraction ───────────────────────────────────────────────
function extractError(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (detail) return typeof detail === 'string' ? detail : JSON.stringify(detail);
    if (err.message) return err.message;
  }
  if (err instanceof Error) return err.message;
  return 'Unknown error occurred';
}

// ── API functions ──────────────────────────────────────────────────
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/api/health');
  return data;
}

export async function fetchModelStatus(): Promise<ModelStatusResponse> {
  const { data } = await api.get<ModelStatusResponse>('/api/model/status');
  return data;
}

export async function predictAudio(audioBlob: Blob, filename: string): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', audioBlob, filename);
  const { data } = await api.post<PredictionResponse>('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return data;
}

export async function fetchDatasetStatus(dataset: string): Promise<DatasetStatusResponse> {
  const { data } = await api.get<DatasetStatusResponse>(`/api/dataset/status?dataset=${dataset}`);
  return data;
}

export async function startTraining(req: TrainRequest): Promise<TrainResponse> {
  const { data } = await api.post<TrainResponse>('/api/train', req);
  return data;
}

export async function fetchTrainStatus(jobId: string): Promise<TrainStatusResponse> {
  const { data } = await api.get<TrainStatusResponse>(`/api/train/status/${jobId}`);
  return data;
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const { data } = await api.get<MetricsResponse>('/api/model/metrics');
  return data;
}

export { extractError };
