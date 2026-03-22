import { BustAngleRadar } from './BustAngleRadar';
import type { BustRecorderResult, BustPhase } from '../hooks/useBustRecorder';
import type { QualityState } from '../types';

interface Props {
  recorder: BustRecorderResult;
  quality:  QualityState;
}

function getFaceGuidance(
  coverage: { neutral: boolean; left: boolean; right: boolean },
  pendingPct: number,
): string {
  const pct = Math.round(pendingPct * 100);
  if (!coverage.neutral) return `Look straight at the camera${pendingPct > 0 ? ` — hold it… ${pct}%` : ''}`;
  if (!coverage.left && !coverage.right) return `Good! Now slowly turn your head to one side`;
  if (!coverage.left) return `Turn to your left side${pendingPct > 0 ? ` — hold it… ${pct}%` : ''}`;
  if (!coverage.right) return `Turn to your right side${pendingPct > 0 ? ` — hold it… ${pct}%` : ''}`;
  return `Face scan complete — ready for bust scan`;
}

function getBustGuidance(
  currentAngle: string | null,
  coverage: Record<string, boolean>,
  pendingPct: number,
): string {
  const pct = Math.round(pendingPct * 100);
  if (!currentAngle) return 'Slowly rotate in front of the camera';
  if (!coverage[currentAngle] && pendingPct > 0) {
    return `${ANGLE_NAMES[currentAngle] ?? currentAngle} detected — hold it… ${pct}%`;
  }
  if (coverage[currentAngle]) {
    return `${ANGLE_NAMES[currentAngle] ?? currentAngle} captured ✓ — keep rotating`;
  }
  return 'Rotate slowly to capture all angles';
}

const ANGLE_NAMES: Record<string, string> = {
  front: 'Front', frontLeft: 'Front-left', left: 'Left side',
  backLeft: 'Back-left', back: 'Back', backRight: 'Back-right',
  right: 'Right side', frontRight: 'Front-right', up: 'Looking up', down: 'Looking down',
};

function phaseLabel(phase: BustPhase): string {
  switch (phase) {
    case 'face-scan':   return 'Step 1 of 2 — Face Scan';
    case 'bust-scan':   return 'Step 2 of 2 — Bust Scan';
    case 'processing':  return 'Processing…';
    case 'done':        return 'Bust Scan Complete';
    default:            return 'Bust Scanner';
  }
}

export function BustScanControls({ recorder, quality }: Props) {
  const {
    phase, faceCoverage, bustCoverage, currentBustAngle, pendingProgress,
    processingProgress, processingMessage, start, advanceToBustScan, stop,
  } = recorder;

  const canStart = quality === 'good' && phase === 'idle';
  const faceComplete = faceCoverage.neutral && (faceCoverage.left || faceCoverage.right);

  return (
    <div style={styles.wrapper}>
      {/* Phase label */}
      <div style={styles.phaseLabel}>{phaseLabel(phase)}</div>

      {/* Phase 1: face scan — show same simple progress as RecordControls */}
      {phase === 'face-scan' && (
        <>
          <div style={styles.barTrack}>
            <div style={{
              ...styles.barFill,
              width: `${Math.round(
                (Object.values(faceCoverage).filter(Boolean).length / 3) * 100
              )}%`,
            }} />
          </div>
          <div style={styles.guidance}>
            <span style={styles.recDot} />
            {getFaceGuidance(faceCoverage, pendingProgress)}
          </div>
          {faceComplete && (
            <button style={{ ...styles.btn, ...styles.btnPrimary }} onClick={advanceToBustScan}>
              Continue to Bust Scan →
            </button>
          )}
        </>
      )}

      {/* Phase 2: bust scan — radar + guidance */}
      {phase === 'bust-scan' && (
        <div style={styles.bustRow}>
          <BustAngleRadar
            coverage={bustCoverage}
            currentAngle={currentBustAngle}
            pendingPct={pendingProgress}
            size={130}
          />
          <div style={styles.bustInfo}>
            <div style={styles.guidance}>
              <span style={styles.recDot} />
              {getBustGuidance(currentBustAngle, bustCoverage, pendingProgress)}
            </div>
            <div style={styles.hint}>
              Rotate slowly in front of the camera.<br/>
              Include head, neck, and upper chest.
            </div>
          </div>
        </div>
      )}

      {/* Phase 3: processing */}
      {phase === 'processing' && (
        <>
          <div style={styles.barTrack}>
            <div style={{ ...styles.barFill, width: `${Math.round(processingProgress * 100)}%` }} />
          </div>
          <div style={{ color: '#818cf8', fontSize: 13 }}>{processingMessage}</div>
        </>
      )}

      {/* Phase 4: done */}
      {phase === 'done' && (
        <div style={{ color: '#4ade80', fontSize: 13 }}>
          ✓ Bust reconstruction complete
        </div>
      )}

      {/* Buttons */}
      <div style={styles.buttonRow}>
        {phase === 'idle' && (
          <button
            style={{ ...styles.btn, ...styles.btnPrimary, opacity: canStart ? 1 : 0.4 }}
            disabled={!canStart}
            onClick={start}
          >
            Start Bust Scan
          </button>
        )}
        {(phase === 'face-scan' || phase === 'bust-scan') && (
          <button style={{ ...styles.btn, ...styles.btnDanger }} onClick={stop}>
            Stop Early
          </button>
        )}
        {phase === 'done' && (
          <button style={{ ...styles.btn, ...styles.btnPrimary }} onClick={start}>
            Rescan
          </button>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    width: 640,
  },
  phaseLabel: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    color: '#555',
  },
  barTrack: {
    width: '100%',
    height: 4,
    borderRadius: 2,
    background: '#1a1a2a',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 2,
    transition: 'width 0.2s ease',
    background: 'linear-gradient(90deg, #818cf8, #c084fc)',
  },
  guidance: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 13,
    color: '#c084fc',
    fontWeight: 500,
  },
  recDot: {
    display: 'inline-block',
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: '#f87171',
    animation: 'pulse 1s infinite',
    flexShrink: 0,
  },
  hint: {
    fontSize: 11,
    color: '#444',
    lineHeight: 1.5,
    marginTop: 4,
  },
  bustRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  bustInfo: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  buttonRow: {
    display: 'flex',
    gap: 8,
    marginTop: 2,
  },
  btn: {
    padding: '8px 20px',
    borderRadius: 8,
    border: 'none',
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 600,
    letterSpacing: '0.02em',
  },
  btnPrimary: {
    background: 'linear-gradient(135deg, #818cf8, #c084fc)',
    color: '#fff',
  },
  btnDanger: {
    background: '#3a1a1a',
    color: '#f87171',
    border: '1px solid #f87171',
  },
};
