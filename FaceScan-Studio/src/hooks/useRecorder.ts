import { useRef, useState, useCallback } from 'react';
import type { RecordFrame, PoseLabel, PoseCoverage } from '../types';
import { estimatePose } from '../lib/poseDetect';

const CAPTURE_INTERVAL_MS = 67;  // ~15 fps
const CONFIRM_FRAMES      = 15;  // frames to hold a pose before confirming (~1 s)

// Minimum poses required for auto-complete
const TARGET_POSES: PoseLabel[] = ['neutral', 'left', 'right'];

type RecordState = 'idle' | 'recording' | 'done';

export interface RecorderResult {
  state: RecordState;
  progress: number;               // 0–1 based on confirmed target poses
  frameCount: number;
  smoothedLandmarks: number[][] | null;
  coverage: PoseCoverage;
  currentPose: PoseLabel | null;
  pendingProgress: number;        // 0–1 for currently-confirming pose
  start: () => void;
  stop: () => void;
}

const EMPTY_COVERAGE: PoseCoverage = {
  neutral: false, left: false, right: false, up: false, down: false,
};

/**
 * Merge confirmed pose frames into a single landmark array.
 * Neutral frames contribute at full weight; non-neutral at 0.3×
 * so the frontal geometry dominates and off-angle frames don't drift landmarks.
 */
function weightedSmooth(confirmed: Map<PoseLabel, RecordFrame[]>): number[][] {
  type Item = { lms: number[][], w: number };
  const items: Item[] = [];

  (confirmed.get('neutral') ?? []).forEach(f => items.push({ lms: f.landmarks, w: 1.0 }));
  (['left', 'right', 'up', 'down'] as PoseLabel[]).forEach(p => {
    (confirmed.get(p) ?? []).forEach(f => items.push({ lms: f.landmarks, w: 0.3 }));
  });

  if (items.length === 0) return [];

  const n = items[0].lms.length;
  const sum = Array.from({ length: n }, () => [0, 0, 0]);
  let totalW = 0;

  for (const { lms, w } of items) {
    for (let i = 0; i < n; i++) {
      sum[i][0] += lms[i][0] * w;
      sum[i][1] += lms[i][1] * w;
      sum[i][2] += lms[i][2] * w;
    }
    totalW += w;
  }

  return sum.map(v => [v[0] / totalW, v[1] / totalW, v[2] / totalW]);
}

export function useRecorder(
  captureFrame: (() => RecordFrame | null) | null,
): RecorderResult {
  const [recState,          setRecState]          = useState<RecordState>('idle');
  const [frameCount,        setFrameCount]        = useState(0);
  const [smoothedLandmarks, setSmoothedLandmarks] = useState<number[][] | null>(null);
  const [coverage,          setCoverage]          = useState<PoseCoverage>(EMPTY_COVERAGE);
  const [currentPose,       setCurrentPose]       = useState<PoseLabel | null>(null);
  const [pendingProgress,   setPendingProgress]   = useState(0);
  const [progress,          setProgress]          = useState(0);

  const intervalRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const confirmedRef    = useRef<Map<PoseLabel, RecordFrame[]>>(new Map());
  const pendingLabelRef = useRef<PoseLabel | null>(null);
  const pendingBufRef   = useRef<RecordFrame[]>([]);
  const totalCountRef   = useRef(0);

  const finalize = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }

    const confirmed = confirmedRef.current;

    // If nothing confirmed yet, salvage the pending buffer
    if (confirmed.size === 0 && pendingBufRef.current.length > 0) {
      confirmed.set(pendingLabelRef.current ?? 'neutral', [...pendingBufRef.current]);
    }

    const smoothed = weightedSmooth(confirmed);
    setSmoothedLandmarks(smoothed.length > 0 ? smoothed : null);
    setRecState('done');
    setCurrentPose(null);
    setPendingProgress(0);
  }, []);

  const start = useCallback(() => {
    if (!captureFrame) return;

    // Reset all state
    confirmedRef.current    = new Map();
    pendingLabelRef.current = null;
    pendingBufRef.current   = [];
    totalCountRef.current   = 0;

    setRecState('recording');
    setSmoothedLandmarks(null);
    setCoverage({ ...EMPTY_COVERAGE });
    setProgress(0);
    setFrameCount(0);
    setCurrentPose(null);
    setPendingProgress(0);

    intervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (!frame) return;

      totalCountRef.current++;
      setFrameCount(totalCountRef.current);

      const pose = estimatePose(frame.landmarks);
      setCurrentPose(pose.label);

      const confirmed = confirmedRef.current;

      if (!confirmed.has(pose.label)) {
        // Accumulate consecutive frames at this pose
        if (pendingLabelRef.current === pose.label) {
          pendingBufRef.current.push(frame);
        } else {
          pendingLabelRef.current = pose.label;
          pendingBufRef.current   = [frame];
        }

        const pct = pendingBufRef.current.length / CONFIRM_FRAMES;
        setPendingProgress(Math.min(pct, 1));

        if (pendingBufRef.current.length >= CONFIRM_FRAMES) {
          // Pose confirmed — snapshot frames and update everything
          confirmed.set(pose.label, [...pendingBufRef.current]);
          pendingLabelRef.current = null;
          pendingBufRef.current   = [];
          setPendingProgress(0);

          // Update coverage display
          const newCov: PoseCoverage = { ...EMPTY_COVERAGE };
          for (const k of confirmed.keys()) newCov[k] = true;
          setCoverage(newCov);

          // Update progress bar (based on target poses)
          const confirmedTargets = TARGET_POSES.filter(p => confirmed.has(p)).length;
          setProgress(Math.min(confirmedTargets / TARGET_POSES.length, 1));

          // Progressively update the 3D preview
          const smoothed = weightedSmooth(confirmed);
          if (smoothed.length > 0) setSmoothedLandmarks(smoothed);

          // Auto-complete when all target poses are confirmed
          if (confirmedTargets >= TARGET_POSES.length) {
            if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
            setRecState('done');
            setCurrentPose(null);
          }
        }
      } else {
        // Already confirmed this pose — reset pending so next new pose starts fresh
        if (pendingLabelRef.current !== null) {
          pendingLabelRef.current = null;
          pendingBufRef.current   = [];
          setPendingProgress(0);
        }
      }
    }, CAPTURE_INTERVAL_MS);
  }, [captureFrame, finalize]);

  const stop = useCallback(() => finalize(), [finalize]);

  return {
    state: recState,
    progress,
    frameCount,
    smoothedLandmarks,
    coverage,
    currentPose,
    pendingProgress,
    start,
    stop,
  };
}
