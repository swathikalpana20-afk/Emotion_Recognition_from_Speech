import React, { useState, useEffect } from 'react';
import { fetchMetrics, fetchModelStatus } from '../services/api';
import type { MetricsResponse, ModelStatusResponse } from '../types/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from 'recharts';

export default function Insights() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [model, setModel] = useState<ModelStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchMetrics(), fetchModelStatus()]).then(([m, ms]) => {
      setMetrics(m);
      setModel(ms);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex-col gap-2" style={{ alignItems: 'center', padding: '4rem' }}>
      <div className="spinner spinner-lg" />
      <div style={{ color: 'var(--text-secondary)' }}>Loading insights...</div>
    </div>
  );

  return (
    <div>
      <div className="mb-3">
        <h1 className="page-title">Model Insights</h1>
        <p className="page-subtitle">Real evaluation metrics and performance analysis</p>
      </div>

      {!metrics?.available && (
        <div className="alert alert-info mb-3">
          ℹ️ No evaluation results available yet. Train the model first to see real metrics here.
        </div>
      )}

      {metrics?.available && (
        <>
          {/* Summary metrics */}
          <div className="grid-4 mb-3">
            {[
              { label: 'Test Accuracy', value: metrics.accuracy, color: 'var(--color-cyan)' },
              { label: 'Macro F1', value: metrics.macro_f1, color: 'var(--color-violet-light)' },
              { label: 'Macro Precision', value: metrics.macro_precision, color: 'var(--color-teal)' },
              { label: 'Macro Recall', value: metrics.macro_recall, color: 'var(--color-emerald)' },
            ].map(({ label, value, color }) => (
              <div key={label} className="card stat-card">
                <div className="stat-value" style={{ color }}>
                  {value != null ? `${(value * 100).toFixed(1)}%` : '—'}
                </div>
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>

          {/* Per-class chart */}
          {metrics.per_class && (
            <div className="card mb-3">
              <div className="section-title">Per-Class F1 Score</div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={Object.entries(metrics.per_class).map(([label, scores]) => ({
                  label,
                  f1: parseFloat((scores['f1-score'] * 100).toFixed(1)),
                  precision: parseFloat((scores.precision * 100).toFixed(1)),
                  recall: parseFloat((scores.recall * 100).toFixed(1)),
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                  <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} unit="%" />
                  <Tooltip
                    contentStyle={{ background: '#131d35', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Legend />
                  <Bar dataKey="f1" fill="#00d4ff" name="F1 %" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="precision" fill="#7c3aed" name="Precision %" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="recall" fill="#14b8a6" name="Recall %" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Confusion matrix */}
          {metrics.confusion_matrix && model?.emotion_labels && (
            <div className="card mb-3">
              <div className="section-title">Confusion Matrix</div>
              <ConfusionMatrix
                matrix={metrics.confusion_matrix}
                labels={model.emotion_labels}
              />
            </div>
          )}

          {metrics.evaluated_at && (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'right' }}>
              Evaluated at: {new Date(metrics.evaluated_at).toLocaleString()}
              {metrics.dataset_name && ` — Dataset: ${metrics.dataset_name.toUpperCase()}`}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ConfusionMatrix({ matrix, labels }: { matrix: number[][], labels: string[] }) {
  const max = Math.max(...matrix.flat());
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: '0.75rem', width: '100%' }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', color: 'var(--text-muted)', textAlign: 'left' }}>True ↓ / Pred →</th>
            {labels.map((l) => (
              <th key={l} style={{ padding: '4px 6px', color: 'var(--color-cyan)', textTransform: 'capitalize', textAlign: 'center' }}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td style={{ padding: '4px 8px', color: 'var(--color-violet-light)', fontWeight: 600, textTransform: 'capitalize', whiteSpace: 'nowrap' }}>
                {labels[i]}
              </td>
              {row.map((val, j) => {
                const intensity = max > 0 ? val / max : 0;
                const isCorrect = i === j;
                return (
                  <td key={j} style={{
                    padding: '4px 6px',
                    textAlign: 'center',
                    background: isCorrect
                      ? `rgba(0, 212, 255, ${intensity * 0.6})`
                      : `rgba(244, 63, 94, ${intensity * 0.4})`,
                    color: intensity > 0.3 ? '#fff' : 'var(--text-muted)',
                    fontFamily: 'var(--font-mono)',
                    borderRadius: '4px',
                  }}>
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
