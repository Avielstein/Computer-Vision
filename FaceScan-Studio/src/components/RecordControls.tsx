import type { QualityState } from '../types';
import type { PoseLabel, PoseCoverage } from '../types';
import type { RecorderResult } from '../hooks/useRecorder';

interface Props {
  recorder: RecorderResult;
  quality: QualityState;
}

// Pose display metadata
const POSE_META: { label: PoseLabel; symbol: string; name: string }[] = [
  { label: 'left',    symbol: '←', name: 'Left'    },
  { label: 'up',      symbol: '↑', name: 'Up'      },
  { label: 'neutral', symbol: '●', name: 'Center'  },
  { label: 'down',    symbol: '↓', name: 'Down'    },
  { label: 'right',   symbol: '→', name: 'Right'   },
];

function getGuidance(
  coverage: PoseCoverage,
  currentPose: PoseLabel | null,
  pendingProgress: number,
): string {
  const pct = Math.round(pendingProgress * 100);

  if (!coverage.neutral) {
    if (currentPose === 'neutral') return `Hold still… ${pct}%`;
    return 'Look straight at the camera to start';
  }

  if (!coverage.left && !coverage.right) {
    if (currentPose === 'left')  return `Left turn detected — hold it… ${pct}%`;
    if (currentPose === 'right') return `Right turn detected — hold it… ${pct}%`;
    return 'Slowly turn your head to one side';
  }

  if (!coverage.left || !coverage.right) {
    const needed = !coverage.left ? 'other side (left)' : 'other side (right)';
    if (currentPose === 'left'  && !coverage.left)  return `Left turn detected — hold it… ${pct}%`;
    if (currentPose === 'right' && !coverage.right) return `Right turn detected — hold it… ${pct}%`;
    if (currentPose === 'neutral') return `Now turn to the ${needed}`;
    return `Hold that position… ${pct}%`;
  }

  // Both sides done
  if (currentPose === 'up'   && !coverage.up)   return `Looking up — hold it… ${pct}%`;
  if (currentPose === 'down' && !coverage.down) return `Looking down — hold it… ${pct}%`;
  return 'All done! Stop or look up/down for extra detail';
}

export function RecordControls({ recorder, quality }: Props) {
  const {
    state, progress, frameCount, coverage, currentPose, pendingProgress, start, stop,
  } = recorder;

  const canRecord = quality === 'good' && state === 'idle';

  const guidanceText = state === 'recording'
    ? getGuidance(coverage, currentPose, pendingProgress)
    : null;

  return (
    <div style={styles.wrapper}>
      {/* Main progress bar — fills as target poses are confirmed */}
      <div style={styles.barTrack}>
        <div style={{
          ...styles.barFill,
          width: `${Math.round(progress * 100)}%`,
          background: state === 'done'
            ? 'linear-gradient(90deg, #4ade80, #22d3ee)'
            : 'linear-gradient(90deg, #818cf8, #c084fc)',
        }} />
      </div>

      {/* Pose coverage indicators */}
      {(state === 'recording' || state === 'done') && (
        <div style={styles.coverageRow}>
          {POSE_META.map(({ label, symbol, name }) => {
            const confirmed = coverage[label];
            const active    = state === 'recording' && currentPose === label && !confirmed;
            return (
              <div
                key={label}
                title={name}
                style={{
                  ...styles.poseDot,
                  color: confirmed ? '#4ade80'
                       : active    ? '#fbbf24'
                       : '#333',
                  borderColor: confirmed ? '#4ade80'
                             : active    ? '#fbbf24'
                             : '#222',
                  animation: active ? 'pulse 0.8s infinite' : 'none',
                }}
              >
                {symbol}
              </div>
            );
          })}

          {/* Pending confirmation mini-bar */}
          {state === 'recording' && currentPose && !coverage[currentPose] && pendingProgress > 0 && (
            <div style={styles.pendingTrack}>
              <div style={{
                ...styles.pendingFill,
                width: `${Math.round(pendingProgress * 100)}%`,
              }} />
            </div>
          )}
        </div>
      )}

      {/* Status / guidance */}
      <div style={styles.statusRow}>
        {state === 'recording' && guidanceText && (
          <span style={styles.guidance}>
            <span style={styles.recDot} />
            {guidanceText}
          </span>
        )}
        {state === 'done' && (
          <span style={{ color: '#4ade80', fontSize: 13 }}>
            ✓ Scan complete — {frameCount} frames · {Object.values(coverage).filter(Boolean).length} poses confirmed
          </span>
        )}
        {state === 'idle' && quality !== 'good' && (
          <span style={{ color: '#f87171', fontSize: 13 }}>
            Position face in frame to enable recording
          </span>
        )}
        {state === 'idle' && quality === 'good' && (
          <span style={{ color: '#555', fontSize: 13 }}>Ready to scan</span>
        )}
      </div>

      {/* Buttons */}
      <div style={styles.buttonRow}>
        {state !== 'recording' ? (
          <button
            style={{ ...styles.btn, ...styles.btnPrimary, opacity: canRecord || state === 'done' ? 1 : 0.4 }}
            disabled={!canRecord && state !== 'done'}
            onClick={start}
          >
            {state === 'done' ? 'Rescan' : 'Start Scan'}
          </button>
        ) : (
          <button style={{ ...styles.btn, ...styles.btnDanger }} onClick={stop}>
            Stop Early
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
    gap: 8,
    width: 640,
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
  },
  coverageRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginTop: 2,
  },
  poseDot: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    border: '1.5px solid',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 13,
    fontWeight: 700,
    transition: 'color 0.2s, border-color 0.2s',
    userSelect: 'none',
  },
  pendingTrack: {
    flex: 1,
    height: 3,
    borderRadius: 2,
    background: '#222',
    overflow: 'hidden',
    marginLeft: 4,
  },
  pendingFill: {
    height: '100%',
    borderRadius: 2,
    background: '#fbbf24',
    transition: 'width 0.1s linear',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    minHeight: 22,
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
  },
  buttonRow: {
    display: 'flex',
    gap: 8,
    marginTop: 4,
  },
  btn: {
    padding: '8px 24px',
    borderRadius: 8,
    border: 'none',
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 600,
    letterSpacing: '0.02em',
    transition: 'opacity 0.15s',
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
