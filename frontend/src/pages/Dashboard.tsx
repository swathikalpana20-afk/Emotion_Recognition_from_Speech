import React from 'react';
import { useFetch } from '../hooks/useApi';
import { fetchModelStatus, fetchMetrics } from '../services/api';
import type { ModelStatusResponse, MetricsResponse } from '../types/api';

const EMOTION_COLORS: Record<string, string> = {
  happy: '#f59e0b',
  sad: '#60a5fa',
  angry: '#f43f5e',
  fearful: '#a78bfa',
  disgust: '#34d399',
  surprised: '#fb923c',
  neutral: '#94a3b8',
  calm: '#38bdf8',
};

export default function Dashboard() {
  const modelState = useFetch<ModelStatusResponse>(fetchModelStatus, []);
  const metricsState = useFetch<MetricsResponse>(fetchMetrics, []);

  const model = modelState.data;
  const metrics = metricsState.data;

  return (
    <div>
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />

      {/* Header */}
      <div className="mb-3">
        <h1 className="page-title text-gradient">EMOTISENSE AI</h1>
        <p className="page-subtitle">
          Speech Emotion Recognition — MFCC + CNN + Bidirectional LSTM
        </p>
      </div>

      {/* Model status banner */}
      {model && (
        <div className={`card mb-3 ${model.available ? 'card-glow-cyan' : ''}`}>
          <div className="flex-row gap-2" style={{ justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <div>
              <div className="flex-row gap-1 mb-1">
                <span className={`badge ${model.available ? 'badge-green' : 'badge-red'}`}>
                  <span className="badge-dot" />
                  {model.available ? 'Model Ready' : 'Model Not Trained'}
                </span>
                {model.dataset_name && (
                  <span className="badge badge-cyan">{model.dataset_name.toUpperCase()}</span>
                )}
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{model.message}</p>
            </div>
            {model.available && (
              <div className="flex-row gap-2" style={{ flexWrap: 'wrap' }}>
                <div className="stat-card">
                  <div className="stat-value" style={{ fontSize: '1.25rem', color: 'var(--color-cyan)' }}>{model.num_classes}</div>
                  <div className="stat-label">Classes</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value" style={{ fontSize: '1.25rem', color: 'var(--color-violet-light)' }}>{model.n_mfcc}</div>
                  <div className="stat-label">MFCC</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value" style={{ fontSize: '1.25rem', color: 'var(--color-teal)' }}>{(model.sample_rate / 1000).toFixed(1)}k</div>
                  <div className="stat-label">Sample Rate</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {modelState.loading && (
        <div className="card mb-3 flex-row gap-2">
          <div className="spinner" />
          <span style={{ color: 'var(--text-secondary)' }}>Loading model status...</span>
        </div>
      )}

      {/* Metrics cards */}
      {metrics?.available && (
        <>
          <h2 className="section-title mt-2">Evaluation Metrics</h2>
          <div className="grid-4 mb-3">
            {[
              { label: 'Accuracy', value: metrics.accuracy, color: 'var(--color-cyan)' },
              { label: 'Macro F1', value: metrics.macro_f1, color: 'var(--color-violet-light)' },
              { label: 'Precision', value: metrics.macro_precision, color: 'var(--color-teal)' },
              { label: 'Recall', value: metrics.macro_recall, color: 'var(--color-emerald)' },
            ].map(({ label, value, color }) => (
              <div key={label} className="card stat-card">
                <div className="stat-value" style={{ color }}>
                  {value != null ? `${(value * 100).toFixed(1)}%` : '—'}
                </div>
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Emotion classes grid */}
      {model?.emotion_labels && (
        <>
          <h2 className="section-title">Supported Emotions</h2>
          <div className="grid-4 mb-3">
            {model.emotion_labels.map((emotion) => (
              <div
                key={emotion}
                className="card"
                style={{
                  borderColor: `${EMOTION_COLORS[emotion] || '#94a3b8'}44`,
                  textAlign: 'center',
                  padding: '1rem',
                }}
              >
                <div style={{ fontSize: '1.75rem', marginBottom: '0.25rem' }}>
                  {emotionEmoji(emotion)}
                </div>
                <div style={{ fontWeight: 600, textTransform: 'capitalize', color: EMOTION_COLORS[emotion] || 'var(--text-primary)' }}>
                  {emotion}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Getting started card */}
      {!model?.available && !modelState.loading && (
        <div className="card card-glass mt-3">
          <h2 className="section-title">🚀 Getting Started</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            <div><span className="badge badge-cyan" style={{ marginRight: '0.5rem' }}>1</span>Download a dataset (RAVDESS, TESS, or EMO-DB) — see Dataset tab</div>
            <div><span className="badge badge-cyan" style={{ marginRight: '0.5rem' }}>2</span>Place it in <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px' }}>emotionai/data/ravdess/</code></div>
            <div><span className="badge badge-cyan" style={{ marginRight: '0.5rem' }}>3</span>Start training via the Training tab or API</div>
            <div><span className="badge badge-cyan" style={{ marginRight: '0.5rem' }}>4</span>Upload audio or record speech to analyze emotions</div>
          </div>
        </div>
      )}

      {/* Scientific disclaimer */}
      <div className="alert alert-info mt-3">
        <span>ℹ️</span>
        <span>
          <strong>Scientific Note:</strong> Emotions inferred from voice are estimates of vocal expression, not reliable proof
          of a person's internal emotional state. Predictions are model estimates only, not psychological diagnoses.
          Softmax confidence scores may be overconfident.
        </span>
      </div>
    </div>
  );
}

function emotionEmoji(emotion: string): string {
  const map: Record<string, string> = {
    happy: '😊', sad: '😢', angry: '😠', fearful: '😨',
    disgust: '🤢', surprised: '😲', neutral: '😐', calm: '😌',
  };
  return map[emotion] || '🎭';
}
