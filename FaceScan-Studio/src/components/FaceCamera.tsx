import { useRef } from 'react';
import type React from 'react';
import { useCamera } from '../hooks/useCamera';
import { useFaceMesh } from '../hooks/useFaceMesh';
import type { QualityState } from '../types';

const QUALITY_LABEL: Record<QualityState, { text: string; color: string }> = {
  good: { text: 'Face Detected', color: '#4ade80' },
  ok:   { text: 'Tracking…',    color: '#facc15' },
  lost: { text: 'No Face',      color: '#f87171' },
};

interface Props {
  onQualityChange?: (q: QualityState) => void;
  onFaceMeshReady?: (captureFrame: () => import('../types').RecordFrame | null) => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

export function FaceCamera({ onQualityChange, onFaceMeshReady, videoRef: externalVideoRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { videoRef, ready, error } = useCamera(externalVideoRef);
  const { quality, captureFrame } = useFaceMesh(videoRef, canvasRef, ready);

  // Propagate captureFrame ref upward once ready
  if (ready && onFaceMeshReady) onFaceMeshReady(captureFrame);
  if (onQualityChange) onQualityChange(quality);

  const badge = QUALITY_LABEL[quality];

  return (
    <div style={styles.wrapper}>
      {error && <div style={styles.error}>{error}</div>}

      {/* Video + canvas stacked */}
      <div style={styles.videoBox}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          width={640}
          height={480}
          style={styles.video}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={styles.overlay}
        />

        {/* Quality badge */}
        <div style={{ ...styles.badge, borderColor: badge.color, color: badge.color }}>
          <span style={{ ...styles.dot, background: badge.color }} />
          {badge.text}
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 12,
  },
  videoBox: {
    position: 'relative',
    width: 640,
    height: 480,
    borderRadius: 12,
    overflow: 'hidden',
    border: '1px solid #2a2a3a',
    background: '#111',
  },
  video: {
    position: 'absolute',
    top: 0, left: 0,
    width: '100%', height: '100%',
    objectFit: 'cover',
    transform: 'scaleX(-1)',  // mirror
  },
  overlay: {
    position: 'absolute',
    top: 0, left: 0,
    width: '100%', height: '100%',
    transform: 'scaleX(-1)',  // mirror to match video
  },
  badge: {
    position: 'absolute',
    top: 12, right: 12,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '4px 10px',
    borderRadius: 20,
    border: '1px solid',
    fontSize: 12,
    fontWeight: 600,
    background: 'rgba(0,0,0,0.55)',
    backdropFilter: 'blur(4px)',
    letterSpacing: '0.03em',
  },
  dot: {
    width: 7, height: 7,
    borderRadius: '50%',
    display: 'inline-block',
  },
  error: {
    color: '#f87171',
    fontSize: 14,
    padding: '8px 16px',
    border: '1px solid #f87171',
    borderRadius: 8,
    background: 'rgba(248,113,113,0.08)',
  },
};
