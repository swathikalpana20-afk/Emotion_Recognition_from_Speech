import React, { useState, useRef, useCallback, useEffect } from 'react';
import { predictAudio, extractError } from '../services/api';
import type { PredictionResponse } from '../types/api';

const EMOTION_COLORS: Record<string, string> = {
  happy: '#f59e0b', sad: '#60a5fa', angry: '#f43f5e', fearful: '#a78bfa',
  disgust: '#34d399', surprised: '#fb923c', neutral: '#94a3b8', calm: '#38bdf8',
};

export default function Analyze() {
  const [activeTab, setActiveTab] = useState<'upload' | 'record'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [micError, setMicError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup URLs
  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  // ── File upload handling ────────────────────────────────────────
  const handleFile = useCallback((f: File) => {
    const allowed = ['.wav', '.mp3', '.ogg', '.flac', '.webm', '.m4a'];
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
    if (!allowed.includes(ext)) {
      setError(`Unsupported format "${ext}". Please use WAV (most reliable), MP3, OGG, or FLAC.`);
      return;
    }
    if (f.size > 25 * 1024 * 1024) {
      setError('File too large. Maximum 25 MB allowed.');
      return;
    }
    setError(null);
    setResult(null);
    setFile(f);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(f));
  }, [audioUrl]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  // ── Microphone recording ────────────────────────────────────────
  const startRecording = async () => {
    setMicError(null);
    setRecordedBlob(null);
    setResult(null);
    chunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: getSupportedMimeType() });
      mediaRecorderRef.current = mr;
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: getSupportedMimeType() });
        setRecordedBlob(blob);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start(100);
      setIsRecording(true);
      setRecordingDuration(0);
      let secs = 0;
      timerRef.current = setInterval(() => {
        secs++;
        setRecordingDuration(secs);
        if (secs >= 30) stopRecording();
      }, 1000);
    } catch (err) {
      setMicError(err instanceof Error ? err.message : 'Microphone access denied. Please allow microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state !== 'inactive') {
      mediaRecorderRef.current?.stop();
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setIsRecording(false);
  };

  // ── Prediction ──────────────────────────────────────────────────
  const runPrediction = async () => {
    let blob: Blob | null = null;
    let filename = 'audio.wav';

    if (activeTab === 'upload' && file) {
      blob = file;
      filename = file.name;
    } else if (activeTab === 'record' && recordedBlob) {
      blob = recordedBlob;
      filename = `recording_${Date.now()}.webm`;
    }

    if (!blob) return;

    setPredicting(true);
    setError(null);
    try {
      const res = await predictAudio(blob, filename);
      setResult(res);
    } catch (err) {
      setError(extractError(err));
    } finally {
      setPredicting(false);
    }
  };

  const canPredict = (activeTab === 'upload' && !!file) || (activeTab === 'record' && !!recordedBlob && !isRecording);

  return (
    <div>
      <div className="mb-3">
        <h1 className="page-title">Audio Analysis Studio</h1>
        <p className="page-subtitle">Upload audio or record your voice to analyze the emotional content</p>
      </div>

      {/* Tab selector */}
      <div className="flex-row gap-1 mb-3">
        {(['upload', 'record'] as const).map((tab) => (
          <button
            key={tab}
            className={`btn ${activeTab === tab ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => { setActiveTab(tab); setResult(null); setError(null); }}
            id={`tab-${tab}`}
          >
            {tab === 'upload' ? '📁 Upload Audio' : '🎙️ Record Speech'}
          </button>
        ))}
      </div>

      <div className="grid-2" style={{ gap: '1.5rem' }}>
        {/* Input panel */}
        <div>
          {activeTab === 'upload' ? (
            <div
              className={`upload-zone ${dragging ? 'drag-over' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onClick={() => fileInputRef.current?.click()}
              id="upload-zone"
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
              aria-label="Upload audio file"
            >
              <div className="upload-zone-icon">🎵</div>
              <div style={{ fontWeight: 700, marginBottom: '0.5rem' }}>
                {file ? file.name : 'Drop audio file here or click to browse'}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                WAV (recommended) · MP3 · OGG · FLAC · Max 25 MB
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.ogg,.flac,.webm,.m4a,audio/*"
                style={{
                  display: 'block',
                  margin: '16px auto 0',
                  maxWidth: '100%',
                  cursor: 'pointer'
                }}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => {
                  const selectedFile = e.currentTarget.files?.[0];
                  if (selectedFile) handleFile(selectedFile);
                  e.currentTarget.value = '';
                }}
                id="file-input"
              />

            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
              <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🎙️</div>
              {micError && <div className="alert alert-error mb-2">{micError}</div>}
              {isRecording ? (
                <div>
                  <div className="flex-row gap-2" style={{ justifyContent: 'center', marginBottom: '1rem' }}>
                    <div className="recording-dot" />
                    <span style={{ fontWeight: 700, color: 'var(--color-rose)' }}>
                      Recording... {recordingDuration}s / 30s
                    </span>
                  </div>
                  <div className="progress-bar-wrap mb-2">
                    <div className="progress-bar-fill" style={{ width: `${(recordingDuration / 30) * 100}%`, background: 'var(--color-rose)' }} />
                  </div>
                  <button className="btn btn-danger" onClick={stopRecording} id="btn-stop-record">
                    ⏹ Stop Recording
                  </button>
                </div>
              ) : (
                <div>
                  {recordedBlob && (
                    <div className="alert alert-success mb-2">
                      ✅ Recording captured ({recordingDuration}s). Ready to analyze.
                    </div>
                  )}
                  <button className="btn btn-primary btn-lg" onClick={startRecording} id="btn-start-record">
                    🎙️ {recordedBlob ? 'Record Again' : 'Start Recording'}
                  </button>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.75rem' }}>
                    Speak clearly for best results. Maximum 30 seconds.
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Audio preview */}
          {audioUrl && (
            <div className="card mt-2">
              <div className="label mb-1">Audio Preview</div>
              <audio
                controls
                src={audioUrl}
                style={{ width: '100%', borderRadius: 'var(--radius-sm)' }}
                id="audio-preview"
              />
            </div>
          )}

          {/* Analyze button */}
          {canPredict && (
            <button
              className="btn btn-primary btn-lg mt-2"
              style={{ width: '100%' }}
              onClick={runPrediction}
              disabled={predicting}
              id="btn-analyze"
            >
              {predicting ? (
                <><div className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }} /> Analyzing...</>
              ) : (
                '🔮 Analyze Emotion'
              )}
            </button>
          )}

          {error && (
            <div className="alert alert-error mt-2">
              <span>⚠️</span> {error}
            </div>
          )}
        </div>

        {/* Results panel */}
        <div>
          {predicting && (
            <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
              <div className="spinner spinner-lg" style={{ margin: '0 auto 1rem' }} />
              <div style={{ fontWeight: 600 }}>Processing audio...</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Extracting MFCC features and running inference
              </div>
            </div>
          )}

          {result && !predicting && (
            <PredictionResult result={result} />
          )}

          {!result && !predicting && (
            <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🎭</div>
              <div style={{ fontWeight: 600 }}>Prediction results will appear here</div>
              <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Upload audio or record speech, then click Analyze
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PredictionResult({ result }: { result: PredictionResponse }) {
  const topEmotion = result.predicted_emotion;
  const color = EMOTION_COLORS[topEmotion] || '#94a3b8';
  const sorted = [...result.probabilities].sort((a, b) => b.probability - a.probability);

  return (
    <div>
      {/* Warnings */}
      {result.silence_warning && (
        <div className="alert alert-warning mb-2">
          ⚠️ The audio appears to be near-silent. Results may be unreliable.
        </div>
      )}
      {result.low_confidence_warning && (
        <div className="alert alert-warning mb-2">
          ⚠️ Low confidence ({(result.confidence * 100).toFixed(1)}%). The model is uncertain about this prediction.
        </div>
      )}

      {/* Main prediction */}
      <div className="card card-glow-cyan mb-2" style={{ borderColor: `${color}55`, textAlign: 'center', padding: '2rem' }}>
        <div style={{ fontSize: '3.5rem', marginBottom: '0.5rem' }}>{emotionEmoji(topEmotion)}</div>
        <div style={{ fontSize: '2rem', fontWeight: 800, textTransform: 'capitalize', color }}>
          {topEmotion}
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
          Predicted emotion
        </div>
        <div style={{ marginTop: '1rem' }}>
          <div className="label mb-1">Model Confidence</div>
          <div className="progress-bar-wrap">
            <div className="progress-bar-fill" style={{ width: `${result.confidence * 100}%`, background: color }} />
          </div>
          <div style={{ textAlign: 'right', fontSize: '0.875rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color, marginTop: '4px' }}>
            {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Probability distribution */}
      <div className="card mb-2">
        <div className="section-title">Probability Distribution</div>
        {sorted.map(({ label, probability }) => (
          <div key={label} className="emotion-bar-row">
            <span className="emotion-bar-label">{label}</span>
            <div className="emotion-bar-track">
              <div
                className="emotion-bar-fill"
                style={{
                  width: `${probability * 100}%`,
                  background: EMOTION_COLORS[label] || '#94a3b8',
                }}


              />
            </div>
            <span
              className="emotion-bar-pct"
              style={{ color: EMOTION_COLORS[label] || 'inherit' }}
            >
              {(probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {/* Metadata */}
      <div className="card">
        <div className="section-title">Analysis Details</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
          {[
            ['Duration', `${result.audio_duration_seconds.toFixed(2)}s`],
            ['MFCC Shape', result.mfcc_shape.join(' × ')],
            ['Model Version', result.model_version],
            ['Preprocessing', result.preprocessing_version],
          ].map(([k, v]) => (
            <div key={k}>
              <div className="label">{k}</div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-cyan)' }}>{v}</div>
            </div>
          ))}
        </div>
        <div className="alert alert-info mt-2" style={{ fontSize: '0.8rem' }}>
          ℹ️ Confidence reflects softmax output — not the probability the person truly feels this emotion.
        </div>
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

function getSupportedMimeType(): string {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}


