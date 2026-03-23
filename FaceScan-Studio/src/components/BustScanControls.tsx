import { BustAngleRadar } from './BustAngleRadar';
import type { BustRecorderResult, ScanMode } from '../hooks/useBustRecorder';
import type { QualityState } from '../types';

interface Props {
  recorder: BustRecorderResult;
  quality:  QualityState;
}

const ANGLE_NAMES: Record<string, string> = {
  front: 'Front', frontLeft: 'Front-left', left: 'Left side',
  backLeft: 'Back-left', back: 'Back', backRight: 'Back-right',
  right: 'Right side', frontRight: 'Front-right', up: 'Up', down: 'Down',
};

const SCAN_MODES: { mode: ScanMode; label: string; desc: string }[] = [
  { mode: 'head',   label: 'Head / Face', desc: 'Face-landmark guided' },
  { mode: 'object', label: 'Object',      desc: 'Optical flow tracking' },
];

function getScanGuidance(
  calibrated: boolean,
  currentAngle: string | null,
  coverage: Record<string, boolean>,
  pendingPct: number,
  trackingMode: 'face' | 'motion',
  scanMode: ScanMode,
): string {
  if (!calibrated) {
    return scanMode === 'object'
      ? 'Place object in view, then start rotating…'
      : 'Face the camera and hold still…';
  }
  if (!currentAngle) return 'Slowly rotate';
  if (coverage[currentAngle]) return `${ANGLE_NAMES[currentAngle] ?? currentAngle} ✓ — keep rotating`;
  if (pendingPct > 0) {
    const pct = Math.round(pendingPct * 100);
    const suffix = trackingMode === 'motion' ? ' — slow down' : '';
    return `${ANGLE_NAMES[currentAngle] ?? currentAngle} — hold it… ${pct}%${suffix}`;
  }
  return `Rotate to ${ANGLE_NAMES[currentAngle] ?? currentAngle}`;
}

export function BustScanControls({ recorder, quality }: Props) {
  const {
    phase, scanMode, setScanMode,
    calibrated, calibrationPct,
    bustCoverage, currentBustAngle, pendingProgress,
    processingProgress, processingMessage,
    trackingMode, canProcess, paused, frameCount,
    start, pause, resume, process, stop, rescanAngle,
  } = recorder;

  const canStart = quality !== 'lost' && (phase === 'idle' || phase === 'done');
  const anglesCaptured = Object.values(bustCoverage).filter(Boolean).length;

  return (
    <div style={styles.wrapper}>

      {/* Scan mode selector — shown when idle or done */}
      {(phase === 'idle' || phase === 'done') && (
        <div style={styles.modeRow}>
          <span style={styles.modeLabel}>Scan mode</span>
          {SCAN_MODES.map(({ mode, label, desc }) => (
            <button
              key={mode}
              title={desc}
              style={{
                ...styles.modeBtn,
                ...(scanMode === mode ? styles.modeBtnActive : {}),
              }}
              onClick={() => setScanMode(mode)}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {/* Scanning */}
      {phase === 'scanning' && (
        <>
          {!calibrated ? (
            <>
              <div style={styles.label}>
                {scanMode === 'object' ? 'Starting…' : 'Calibrating — look straight at the camera'}
              </div>
              <div style={styles.barTrack}>
                <div style={{ ...styles.barFill, width: `${Math.round(calibrationPct * 100)}%` }} />
              </div>
            </>
          ) : (
            <>
              <div style={styles.bustRow}>
                <BustAngleRadar
                  coverage={bustCoverage}
                  currentAngle={currentBustAngle}
                  pendingPct={pendingProgress}
                  size={130}
                  onRescan={rescanAngle}
                />
                <div style={styles.bustInfo}>
                  <div style={styles.guidance}>
                    <span style={{
                      ...styles.dot,
                      background: trackingMode === 'face' ? '#4ade80' : '#facc15',
                    }} />
                    {getScanGuidance(
                      calibrated, currentBustAngle, bustCoverage,
                      pendingProgress, trackingMode, scanMode,
                    )}
                  </div>
                  <div style={styles.trackBadge}>
                    {trackingMode === 'face' ? '● Face tracked' : '◌ Motion tracked — turn slowly'}
                  </div>
                  <div style={styles.frameCounter}>
                    {frameCount} frame{frameCount !== 1 ? 's' : ''} captured
                    {anglesCaptured > 0 && ` · ${anglesCaptured} angle${anglesCaptured !== 1 ? 's' : ''}`}
                  </div>
                  <div style={styles.hint}>
                    Rotate slowly for full coverage. Click a green dot to re-scan an angle.
                  </div>
                </div>
              </div>
            </>
          )}

          <div style={styles.btnRow}>
            {canProcess && (
              <button style={{ ...styles.btn, ...styles.btnPrimary }} onClick={process}>
                Process ({frameCount})
              </button>
            )}
            <button
              style={{ ...styles.btn, ...styles.btnSecondary }}
              onClick={paused ? resume : pause}
            >
              {paused ? '▶ Resume' : '⏸ Pause'}
            </button>
            <button style={{ ...styles.btn, ...styles.btnDanger }} onClick={stop}>
              Cancel
            </button>
          </div>
        </>
      )}

      {/* Processing */}
      {phase === 'processing' && (
        <>
          <div style={styles.label}>Reconstructing {frameCount} frames…</div>
          <div style={styles.barTrack}>
            <div style={{ ...styles.barFill, width: `${Math.round(processingProgress * 100)}%` }} />
          </div>
          <div style={styles.progressMsg}>{processingMessage}</div>
        </>
      )}

      {/* Done */}
      {phase === 'done' && (
        <div style={styles.doneRow}>
          <span style={{ color: '#4ade80', fontSize: 13 }}>✓ Complete</span>
        </div>
      )}

      {/* Start button */}
      {(phase === 'idle' || phase === 'done') && (
        <button
          style={{ ...styles.btn, ...styles.btnPrimary, opacity: canStart ? 1 : 0.4 }}
          disabled={!canStart}
          onClick={start}
        >
          {phase === 'done' ? 'Rescan' : 'Start Scan'}
        </button>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrapper:       { display: 'flex', flexDirection: 'column', gap: 10, width: 640 },
  label:         { fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#555' },
  barTrack:      { width: '100%', height: 4, borderRadius: 2, background: '#1a1a2a', overflow: 'hidden' },
  barFill:       { height: '100%', borderRadius: 2, transition: 'width 0.2s ease', background: 'linear-gradient(90deg, #818cf8, #c084fc)' },
  bustRow:       { display: 'flex', alignItems: 'center', gap: 16 },
  bustInfo:      { flex: 1, display: 'flex', flexDirection: 'column', gap: 6 },
  guidance:      { display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: '#c084fc', fontWeight: 500 },
  dot:           { display: 'inline-block', width: 7, height: 7, borderRadius: '50%', flexShrink: 0 },
  trackBadge:    { fontSize: 11, fontWeight: 600, color: '#555' },
  frameCounter:  { fontSize: 12, color: '#818cf8', fontWeight: 500 },
  hint:          { fontSize: 11, color: '#444', lineHeight: 1.5 },
  progressMsg:   { fontSize: 13, color: '#818cf8' },
  doneRow:       { display: 'flex', alignItems: 'center', gap: 8 },
  btnRow:        { display: 'flex', gap: 8, marginTop: 2 },
  btn:           { padding: '8px 20px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600 },
  btnPrimary:    { background: 'linear-gradient(135deg, #818cf8, #c084fc)', color: '#fff' },
  btnSecondary:  { background: '#1a1a2a', color: '#818cf8', border: '1px solid #818cf8' },
  btnDanger:     { background: '#3a1a1a', color: '#f87171', border: '1px solid #f87171' },
  modeRow:       { display: 'flex', alignItems: 'center', gap: 8 },
  modeLabel:     { fontSize: 11, color: '#555', fontWeight: 600, marginRight: 2 },
  modeBtn:       { padding: '4px 12px', borderRadius: 6, border: '1px solid #2a2a3a', background: '#0d0d15', color: '#555', fontSize: 12, cursor: 'pointer', fontWeight: 600 },
  modeBtnActive: { border: '1px solid #c084fc', color: '#c084fc', background: '#1a0a2a' },
};
