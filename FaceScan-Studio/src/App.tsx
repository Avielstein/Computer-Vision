import { useState, useRef, useCallback } from 'react';
import { FaceCamera } from './components/FaceCamera';
import { RecordControls } from './components/RecordControls';
import { Preview3D } from './components/Preview3D';
import { PartToggles } from './components/PartToggles';
import { useRecorder } from './hooks/useRecorder';
import { DEFAULT_PART_VISIBILITY } from './types';
import type { QualityState, RecordFrame, PartVisibility } from './types';

export default function App() {
  const [quality, setQuality] = useState<QualityState>('lost');
  const [partVisibility, setPartVisibility] = useState<PartVisibility>(DEFAULT_PART_VISIBILITY);
  const captureFrameRef = useRef<(() => RecordFrame | null) | null>(null);

  const captureFrame = useCallback(() => {
    return captureFrameRef.current ? captureFrameRef.current() : null;
  }, []);

  const recorder = useRecorder(captureFrame);

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <h1 style={styles.title}>FaceScan Studio</h1>
        <span style={styles.subtitle}>Real-time face segmentation &amp; 3D scanning</span>
      </header>

      <main style={styles.main}>
        {/* Left panel: camera + record controls */}
        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>Live Feed</h2>
          <FaceCamera
            onQualityChange={setQuality}
            onFaceMeshReady={fn => { captureFrameRef.current = fn; }}
          />
          <RecordControls recorder={recorder} quality={quality} />
        </section>

        {/* Right panel: 3D preview + part toggles */}
        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>3D Preview</h2>
          <Preview3D
            landmarks={recorder.smoothedLandmarks}
            partVisibility={partVisibility}
          />
          <PartToggles
            visibility={partVisibility}
            onChange={setPartVisibility}
            disabled={!recorder.smoothedLandmarks}
          />
        </section>
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: '#0a0a0f',
    color: '#e0e0e8',
  },
  header: {
    padding: '20px 32px 12px',
    borderBottom: '1px solid #1a1a2a',
    display: 'flex',
    alignItems: 'baseline',
    gap: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 700,
    letterSpacing: '-0.02em',
    background: 'linear-gradient(135deg, #818cf8, #c084fc)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  subtitle: {
    fontSize: 13,
    color: '#555',
  },
  main: {
    flex: 1,
    display: 'flex',
    gap: 24,
    padding: 24,
    alignItems: 'flex-start',
  },
  panel: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    flex: 1,
    minWidth: 0,
  },
  panelTitle: {
    fontSize: 13,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: '#666',
  },
};
