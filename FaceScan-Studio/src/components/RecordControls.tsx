import type { QualityState } from '../types';
import type { RecorderResult } from '../hooks/useRecorder';

const GUIDANCE_STEPS = [
  'Look straight ahead',
  'Turn head left',
  'Turn head right',
  'Look up',
  'Look down',
];

interface Props {
  recorder: RecorderResult;
  quality: QualityState;
}

export function RecordControls({ recorder, quality }: Props) {
  const { state, progress, frameCount, start, stop } = recorder;

  const guidanceIndex = Math.min(
    Math.floor(progress * GUIDANCE_STEPS.length),
    GUIDANCE_STEPS.length - 1
  );
  const guidanceText = GUIDANCE_STEPS[guidanceIndex];

  const canRecord = quality === 'good' && state === 'idle';

  return (
    <div style={styles.wrapper}>
      {/* Progress bar */}
      <div style={styles.barTrack}>
        <div
          style={{
            ...styles.barFill,
            width: `${Math.round(progress * 100)}%`,
            background: state === 'done'
              ? 'linear-gradient(90deg, #4ade80, #22d3ee)'
              : 'linear-gradient(90deg, #818cf8, #c084fc)',
          }}
        />
      </div>

      {/* Status row */}
      <div style={styles.statusRow}>
        {state === 'recording' && (
          <span style={styles.guidance}>
            <span style={styles.recDot} /> {guidanceText}
          </span>
        )}
        {state === 'done' && (
          <span style={{ color: '#4ade80', fontSize: 13 }}>
            ✓ Scan complete — {frameCount} frames captured
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

        <span style={{ color: '#444', fontSize: 12, marginLeft: 'auto' }}>
          {state === 'recording' ? `${Math.round(progress * 100)}%` : ''}
        </span>
      </div>

      {/* Buttons */}
      <div style={styles.buttonRow}>
        {state !== 'recording' ? (
          <button
            style={{ ...styles.btn, ...styles.btnPrimary, opacity: canRecord || state === 'done' ? 1 : 0.4 }}
            disabled={!canRecord && state !== 'done'}
            onClick={state === 'done' ? start : start}
          >
            {state === 'done' ? 'Rescan' : 'Record'}
          </button>
        ) : (
          <button style={{ ...styles.btn, ...styles.btnDanger }} onClick={stop}>
            Stop
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
