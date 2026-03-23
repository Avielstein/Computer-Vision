import { useState, useRef, useCallback } from 'react';
import { FaceCamera, type OverlayMode } from './components/FaceCamera';
import { BustScanControls } from './components/BustScanControls';
import { Preview3D, type RenderMode } from './components/Preview3D';
import { useBustRecorder } from './hooks/useBustRecorder';
import { useSegmentation } from './hooks/useSegmentation';
import type { QualityState, RecordFrame } from './types';
import { exportSTL } from './lib/exportSTL';

const RENDER_MODES: { mode: RenderMode; label: string }[] = [
  { mode: 'mono',      label: 'Mono'    },
  { mode: 'depth',     label: 'Depth'   },
  { mode: 'segmented', label: 'Regions' },
  { mode: 'texture',   label: 'Texture' },
];

const OVERLAY_MODES: { mode: OverlayMode; label: string }[] = [
  { mode: 'none',       label: 'Off'        },
  { mode: 'regions',    label: 'Regions'    },
  { mode: 'confidence', label: 'Confidence' },
  { mode: 'mesh',       label: 'Mesh'       },
];

export default function App() {
  const [quality,      setQuality]      = useState<QualityState>('lost');
  const [cameraReady,  setCameraReady]  = useState(false);
  const [renderMode,   setRenderMode]   = useState<RenderMode>('mono');
  const [overlayMode,  setOverlayMode]  = useState<OverlayMode>('regions');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureFrameRef = useRef<(() => RecordFrame | null) | null>(null);

  const captureFrame = useCallback(() => {
    return captureFrameRef.current ? captureFrameRef.current() : null;
  }, []);

  const segmentation = useSegmentation(videoRef, cameraReady);
  const bustRecorder = useBustRecorder(
    captureFrame,
    segmentation.captureMask,
    segmentation.captureColorFrame,
    videoRef,
  );

  const handleQualityChange = useCallback((q: QualityState) => {
    setQuality(q);
    setCameraReady(q !== 'lost');
  }, []);

  const handleExportSTL = useCallback(() => {
    if (!bustRecorder.bustGeometry) return;
    exportSTL(bustRecorder.bustGeometry, 'bust.stl');
  }, [bustRecorder.bustGeometry]);

  const displayLandmarks = bustRecorder.faceLandmarks;

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <h1 style={styles.title}>FaceScan Studio</h1>
        <span style={styles.subtitle}>3D bust scanning</span>
      </header>

      <main style={styles.main}>
        {/* Left panel: camera + controls */}
        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>Live Feed</h2>
          <FaceCamera
            videoRef={videoRef}
            onQualityChange={handleQualityChange}
            onFaceMeshReady={fn => { captureFrameRef.current = fn; }}
            overlayMode={overlayMode}
          />
          <div style={styles.modeRow}>
            {OVERLAY_MODES.map(({ mode, label }) => (
              <button
                key={mode}
                style={{ ...styles.modeBtn, ...(overlayMode === mode ? styles.modeBtnActive : {}) }}
                onClick={() => setOverlayMode(mode)}
              >
                {label}
              </button>
            ))}
          </div>
          <BustScanControls recorder={bustRecorder} quality={quality} />
        </section>

        {/* Right panel: 3D preview + controls */}
        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>3D Preview</h2>
          <Preview3D
            landmarks={displayLandmarks}
            bustGeometry={bustRecorder.bustGeometry}
            renderMode={renderMode}
          />
          <div style={styles.modeRow}>
            {RENDER_MODES.map(({ mode, label }) => (
              <button
                key={mode}
                style={{
                  ...styles.modeBtn,
                  ...(renderMode === mode ? styles.modeBtnActive : {}),
                }}
                onClick={() => setRenderMode(mode)}
              >
                {label}
              </button>
            ))}
          </div>
          {bustRecorder.phase === 'done' && bustRecorder.bustGeometry && (
            <div style={styles.exportRow}>
              <button onClick={handleExportSTL} style={styles.exportBtn}>
                ⬇ Export Bust STL
              </button>
            </div>
          )}
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
  modeRow: {
    display: 'flex',
    gap: 6,
  },
  modeBtn: {
    padding: '5px 14px',
    borderRadius: 6,
    border: '1px solid #2a2a3a',
    background: '#0d0d15',
    color: '#555',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    letterSpacing: '0.04em',
  },
  modeBtnActive: {
    border: '1px solid #818cf8',
    color: '#818cf8',
    background: '#1a1a2a',
  },
  exportRow: {
    display: 'flex',
    gap: 8,
  },
  exportBtn: {
    padding: '10px 20px',
    background: 'linear-gradient(135deg, #818cf8, #c084fc)',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  },
};
