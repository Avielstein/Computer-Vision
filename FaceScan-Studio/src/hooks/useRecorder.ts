import { useRef, useState, useCallback } from 'react';
import type { RecordFrame } from '../types';

const DURATION_MS = 5000;
const CAPTURE_INTERVAL_MS = 67; // ~15fps

type RecordState = 'idle' | 'recording' | 'done';

export interface RecorderResult {
  state: RecordState;
  progress: number;               // 0–1
  frameCount: number;
  smoothedLandmarks: number[][] | null;  // 468 × [x,y,z], null until done
  start: () => void;
  stop: () => void;
}

function smoothLandmarks(frames: RecordFrame[]): number[][] {
  if (frames.length === 0) return [];
  const n = frames[0].landmarks.length;  // 468
  const result: number[][] = Array.from({ length: n }, () => [0, 0, 0]);

  for (const frame of frames) {
    for (let i = 0; i < n; i++) {
      result[i][0] += frame.landmarks[i][0];
      result[i][1] += frame.landmarks[i][1];
      result[i][2] += frame.landmarks[i][2];
    }
  }

  const count = frames.length;
  for (let i = 0; i < n; i++) {
    result[i][0] /= count;
    result[i][1] /= count;
    result[i][2] /= count;
  }

  return result;
}

export function useRecorder(
  captureFrame: (() => RecordFrame | null) | null
): RecorderResult {
  const [recState, setRecState] = useState<RecordState>('idle');
  const [progress, setProgress] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [smoothedLandmarks, setSmoothedLandmarks] = useState<number[][] | null>(null);

  const framesRef = useRef<RecordFrame[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const finish = useCallback((frames: RecordFrame[]) => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    const smoothed = smoothLandmarks(frames);
    setSmoothedLandmarks(smoothed);
    setProgress(1);
    setRecState('done');
  }, []);

  const start = useCallback(() => {
    if (!captureFrame) return;
    framesRef.current = [];
    setSmoothedLandmarks(null);
    setProgress(0);
    setFrameCount(0);
    setRecState('recording');
    startTimeRef.current = performance.now();

    intervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        framesRef.current.push(frame);
        setFrameCount(framesRef.current.length);
      }

      const elapsed = performance.now() - startTimeRef.current;
      const p = Math.min(elapsed / DURATION_MS, 1);
      setProgress(p);

      if (elapsed >= DURATION_MS) {
        finish(framesRef.current);
      }
    }, CAPTURE_INTERVAL_MS);
  }, [captureFrame, finish]);

  const stop = useCallback(() => {
    finish(framesRef.current);
  }, [finish]);

  return { state: recState, progress, frameCount, smoothedLandmarks, start, stop };
}
