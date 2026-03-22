import { useState, useRef, useCallback } from 'react';
import { FaceCamera } from './components/FaceCamera';
import { RecordControls } from './components/RecordControls';
import { BustScanControls } from './components/BustScanControls';
import { Preview3D } from './components/Preview3D';
import { PartToggles } from './components/PartToggles';
import { useRecorder } from './hooks/useRecorder';
import { useBustRecorder } from './hooks/useBustRecorder';
import { useSegmentation } from './hooks/useSegmentation';
import { DEFAULT_PART_VISIBILITY } from './types';
import type { QualityState, RecordFrame, PartVisibility } from './types';
import { buildPrintableBust } from './lib/meshBuilder';
import { exportSTL } from './lib/exportSTL';

type AppMode = 'face' | 'bust';

export default function App() {
  const [mode, setMode] = useState<AppMode>('face');
  const [quality, setQuality] = useState<QualityState>('lost');
  const [cameraReady, setCameraReady] = useState(false);
  const [partVisibility, setPartVisibility] = useState<PartVisibility>(DEFAULT_PART_VISIBILITY);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureFrameRef = useRef<(() => RecordFrame | null) | null>(null);

  const captureFrame = useCallback(() => {
    return captureFrameRef.current ? captureFrameRef.current() : null;
  }, []);

  const segmentation = useSegmentation(videoRef, cameraReady);
  const recorder = useRecorder(captureFrame);
  const bustRecorder = useBustRecorder(captureFrame, segmentation.captureMask);

  const handleQualityChange = useCallback((q: QualityState) => {
    setQuality(q);
    setCameraReady(q !== 'lost');
  }, []);

  // Face-only STL export
  const handleExportFaceSTL = useCallback(() => {
    if (!recorder.smoothedLandmarks) return;
    const geometry = buildPrintableBust(recorder.smoothedLandmarks);
    exportSTL(geometry, 'face-bust.stl');
  }, [recorder.smoothedLandmarks]);

  // Full bust STL export
  const handleExportBustSTL = useCallback(() => {
    if (!bustRecorder.bustGeometry) return;
    exportSTL(bustRecorder.bustGeometry, 'full-bust.stl');
  }, [bustRecorder.bustGeometry]);

  const displayLandmarks = mode === 'bust'
    ? (bustRecorder.faceLandmarks ?? recorder.smoothedLandmarks)
    : recorder.smoothedLandmarks;

  const bustGeometry = mode === 'bust' ? bustRecorder.bustGeometry : null;

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.title}>FaceScan Studio</h1>
          <span style={styles.subtitle}>Real-time face segmentation &amp; 3D scanning</span>
        </div>
        <div style={styles.modeToggle}>
          <button
            style={{ ...styles.modeBtn, ...(mode === 'face' ? styles.modeBtnActive : {}) }}
            onClick={() => setMode('face')}
          >
            Face Scan
          </button>
          <button
            style={{ ...styles.modeBtn, ...(mode === 'bust' ? styles.modeBtnActive : {}) }}
            onClick={() => setMode('bust')}
          >
            Full Bust
          </button>
        </div>
      </header>

      <main style={styles.main}>
        {/* Left panel: camera + controls */}
        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>Live Feed</h2>
          <FaceCamera
            videoRef={videoRef}
            onQualityChange={handleQualityChange}
            onFaceMeshReady={fn => { captureFrameRef.current = fn; }}
          />
          {mode === 'face' ? (
            <RecordControls recorder={recorder} quality={quality} />
          ) : (
            <BustScanControls recorder={bustRecorder} quality={quality} />
          )}
        </section>

        {/* Right panel: 3D preview + controls */}
        <section style={styles.panel}>
          <h2 style={styles.panelTitle}>3D Preview</h2>
          <Preview3D
            landmarks={displayLandmarks}
            partVisibility={partVisibility}
            bustGeometry={bustGeometry}
          />
          <PartToggles
            visibility={partVisibility}
            onChange={setPartVisibility}
            disabled={!displayLandmarks}
          />
          <div style={styles.exportRow}>
            {mode === 'face' && recorder.state === 'done' && (
              <button onClick={handleExportFaceSTL} style={styles.exportBtn}>
                ⬇ Export Face STL
              </button>
            )}
            {mode === 'bust' && bustRecorder.phase === 'done' && bustRecorder.bustGeometry && (
              <button onClick={handleExportBustSTL} style={styles.exportBtn}>
                ⬇ Export Full Bust STL
              </button>
            )}
            {mode === 'bust' && bustRecorder.faceLandmarks && (
              <button onClick={handleExportFaceSTL} style={{ ...styles.exportBtn, ...styles.exportBtnSecondary }}>
                ⬇ Face Only STL
              </button>
            )}
          </div>
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
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: {
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
  modeToggle: {
    display: 'flex',
    gap: 4,
    background: '#1a1a2a',
    padding: 4,
    borderRadius: 10,
  },
  modeBtn: {
    padding: '6px 18px',
    borderRadius: 7,
    border: 'none',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 600,
    background: 'transparent',
    color: '#666',
    transition: 'all 0.15s',
  },
  modeBtnActive: {
    background: 'linear-gradient(135deg, #818cf8, #c084fc)',
    color: '#fff',
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
  exportRow: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
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
  exportBtnSecondary: {
    background: '#1a1a2a',
    color: '#818cf8',
    border: '1px solid #818cf8',
  },
};
