// EMOTISENSE AI — API types matching backend schemas
export interface HealthResponse {
  status: string;
  version: string;
  dependencies: Record<string, string>;
}

export interface ModelStatusResponse {
  available: boolean;
  model_version: string | null;
  preprocessing_version: string | null;
  dataset_name: string | null;
  emotion_labels: string[];
  num_classes: number;
  sample_rate: number;
  n_mfcc: number;
  use_delta: boolean;
  use_delta2: boolean;
  feature_dim: number;
  last_trained: string | null;
  message: string;
}

export interface EmotionProbability {
  label: string;
  probability: number;
}

export interface PredictionResponse {
  predicted_emotion: string;
  confidence: number;
  probabilities: EmotionProbability[];
  model_version: string;
  preprocessing_version: string;
  low_confidence_warning: boolean;
  silence_warning: boolean;
  mfcc_shape: number[];
  audio_duration_seconds: number;
  message?: string;
}

export interface DatasetStatusResponse {
  dataset_name: string;
  available: boolean;
  total_samples: number;
  class_counts: Record<string, number>;
  missing_classes: string[];
  message: string;
}

export interface TrainRequest {
  dataset: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  model_type?: string;
}

export interface TrainResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface TrainStatusResponse {
  job_id: string;
  status: string;
  progress_percent: number;
  current_epoch: number;
  total_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  message: string;
  error: string | null;
}

export interface MetricsResponse {
  available: boolean;
  accuracy: number | null;
  macro_precision: number | null;
  macro_recall: number | null;
  macro_f1: number | null;
  per_class: Record<string, Record<string, number>> | null;
  confusion_matrix: number[][] | null;
  dataset_name: string | null;
  evaluated_at: string | null;
  message: string;
}
