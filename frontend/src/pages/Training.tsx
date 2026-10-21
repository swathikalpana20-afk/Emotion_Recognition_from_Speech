import React, { useState, useEffect } from 'react';
import { startTraining, fetchTrainStatus, fetchDatasetStatus } from '../services/api';
import type { TrainStatusResponse, DatasetStatusResponse } from '../types/api';

const DATASETS = ['ravdess', 'tess', 'emodb'];

export default function Training() {
  const [dataset, setDataset] = useState('ravdess');
  const [epochs, setEpochs] = useState(50);
  const [modelType, setModelType] = useState('cnn_bilstm');
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<TrainStatusResponse | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    const poll = setInterval(async () => {
      try {
        const s = await fetchTrainStatus(jobId);
        setJobStatus(s);
        if (s.status === 'completed' || s.status === 'failed') clearInterval(poll);
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(poll);
  }, [jobId]);

  // Fetch dataset info on dataset change
  useEffect(() => {
    setDatasetInfo(null);
    fetchDatasetStatus(dataset).then(setDatasetInfo).catch(() => setDatasetInfo(null));
  }, [dataset]);

  const handleTrain = async () => {
    setError(null);
    setLoading(true);
    setJobId(null);
    setJobStatus(null);
    try {
      const res = await startTraining({ dataset, epochs, model_type: modelType });
      setJobId(res.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const isRunning = jobStatus?.status === 'running' || jobStatus?.status === 'pending';

  return (
    <div>
      <div className="mb-3">
        <h1 className="page-title">Training</h1>
        <p className="page-subtitle">Train the CNN + BiLSTM model on a speech emotion dataset</p>
      </div>

      <div className="grid-2" style={{ gap: '1.5rem' }}>
        {/* Config panel */}
        <div>
          <div className="card mb-2">
            <div className="section-title">Training Configuration</div>

            <div className="mb-2">
              <label className="label" htmlFor="sel-dataset">Dataset</label>
              <select
                id="sel-dataset"
                className="input mt-1"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                disabled={isRunning}
              >
                {DATASETS.map((d) => (
                  <option key={d} value={d}>{d.toUpperCase()}</option>
                ))}
              </select>
            </div>

            <div className="mb-2">
              <label className="label" htmlFor="sel-model">Model Architecture</label>
              <select
                id="sel-model"
                className="input mt-1"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                disabled={isRunning}
              >
                <option value="cnn_bilstm">CNN + Bidirectional LSTM (Recommended)</option>
                <option value="cnn_baseline">CNN Baseline</option>
              </select>
            </div>

            <div className="mb-2">
              <label className="label" htmlFor="inp-epochs">Epochs</label>
              <input
                id="inp-epochs"
                type="number"
                className="input mt-1"
                value={epochs}
                min={1} max={300}
                onChange={(e) => setEpochs(Number(e.target.value))}
                disabled={isRunning}
              />
            </div>

            {error && <div className="alert alert-error mb-2">⚠️ {error}</div>}

            <button
              className="btn btn-primary"
              style={{ width: '100%' }}
              onClick={handleTrain}
              disabled={loading || isRunning}
              id="btn-start-training"
            >
              {loading || isRunning ? (
                <><div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} /> Training in progress...</>
              ) : '🚀 Start Training'}
            </button>
          </div>

          {/* Dataset status */}
          {datasetInfo && (
            <div className="card">
              <div className="section-title">Dataset Status</div>
              <div className="flex-row gap-1 mb-2">
                <span className={`badge ${datasetInfo.available ? 'badge-green' : 'badge-red'}`}>
                  <span className="badge-dot" />
                  {datasetInfo.available ? 'Available' : 'Not Found'}
                </span>
                <span className="badge badge-cyan">{datasetInfo.total_samples} samples</span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{datasetInfo.message}</p>
              {!datasetInfo.available && (
                <div className="alert alert-warning mt-2" style={{ fontSize: '0.8rem' }}>
                  Download the {dataset.toUpperCase()} dataset and place it at:
                  <code style={{ display: 'block', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
                    emotionai/data/{dataset}/
                  </code>
                </div>
              )}
              {datasetInfo.available && Object.entries(datasetInfo.class_counts).length > 0 && (
                <div className="mt-2">
                  <div className="label mb-1">Class Distribution</div>
                  {Object.entries(datasetInfo.class_counts).sort((a, b) => b[1] - a[1]).map(([label, count]) => (
                    <div key={label} className="emotion-bar-row">
                      <span className="emotion-bar-label">{label}</span>
                      <div className="emotion-bar-track">
                        <div className="emotion-bar-fill" style={{
                          width: `${(count / datasetInfo.total_samples) * 100}%`,
                          background: 'var(--color-cyan)',
                        }} />
                      </div>
                      <span className="emotion-bar-pct">{count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Progress panel */}
        <div>
          {jobStatus ? (
            <div className="card">
              <div className="flex-row gap-2 mb-2" style={{ justifyContent: 'space-between' }}>
                <div className="section-title" style={{ margin: 0 }}>
                  Job: <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-cyan)' }}>{jobStatus.job_id}</code>
                </div>
                <span className={`badge ${
                  jobStatus.status === 'completed' ? 'badge-green' :
                  jobStatus.status === 'failed' ? 'badge-red' :
                  'badge-amber'
                }`}>
                  <span className="badge-dot" />
                  {jobStatus.status}
                </span>
              </div>

              <div className="mb-2">
                <div className="label mb-1">
                  Progress — Epoch {jobStatus.current_epoch} / {jobStatus.total_epochs}
                </div>
                <div className="progress-bar-wrap">
                  <div className="progress-bar-fill" style={{ width: `${jobStatus.progress_percent}%` }} />
                </div>
                <div style={{ textAlign: 'right', fontSize: '0.8rem', color: 'var(--color-cyan)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
                  {jobStatus.progress_percent.toFixed(1)}%
                </div>
              </div>

              {/* Live metrics */}
              {jobStatus.train_loss != null && (
                <div className="grid-3 mb-2">
                  {[
                    ['Train Loss', jobStatus.train_loss?.toFixed(4), 'var(--color-cyan)'],
                    ['Val Loss', jobStatus.val_loss?.toFixed(4), 'var(--color-violet-light)'],
                    ['Val Acc', jobStatus.val_accuracy != null ? `${(jobStatus.val_accuracy * 100).toFixed(1)}%` : '—', 'var(--color-emerald)'],
                  ].map(([label, value, color]) => (
                    <div key={label as string} className="card stat-card" style={{ padding: '0.75rem' }}>
                      <div className="stat-value" style={{ fontSize: '1.1rem', color: color as string }}>{value}</div>
                      <div className="stat-label">{label}</div>
                    </div>
                  ))}
                </div>
              )}

              <div style={{
                background: 'rgba(0,0,0,0.3)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.75rem',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.8rem',
                color: 'var(--text-secondary)',
                wordBreak: 'break-word',
              }}>
                {jobStatus.message}
              </div>

              {jobStatus.error && (
                <div className="alert alert-error mt-2">⚠️ {jobStatus.error}</div>
              )}

              {jobStatus.status === 'completed' && (
                <div className="alert alert-success mt-2">
                  ✅ Training complete! Switch to Insights tab to view evaluation metrics.
                </div>
              )}
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🧠</div>
              <div style={{ fontWeight: 600 }}>Training progress will appear here</div>
              <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Configure training and click Start Training
              </div>
            </div>
          )}

          {/* Architecture info */}
          <div className="card mt-2">
            <div className="section-title">Architecture Overview</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
              {[
                ['Input', 'MFCC + delta + delta² features [T × 120]'],
                ['CNN Block', '2× Conv2D + BatchNorm + MaxPool + Dropout'],
                ['Projection', 'Linear → LSTM input dimension'],
                ['BiLSTM', '2-layer Bidirectional LSTM (256 hidden)'],
                ['Head', 'Linear → ReLU → Dropout → Linear → 8 classes'],
                ['Loss', 'CrossEntropyLoss + AdamW + CosineAnnealing'],
              ].map(([k, v]) => (
                <div key={k} className="flex-row gap-2" style={{ marginBottom: '4px' }}>
                  <span style={{ color: 'var(--color-cyan)', minWidth: '80px', fontWeight: 600 }}>{k}</span>
                  <span>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
