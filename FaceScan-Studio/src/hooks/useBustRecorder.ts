/**
 * Two-phase bust recorder:
 *   Phase 1 (face-scan): same pose-driven logic as useRecorder
 *   Phase 2 (bust-scan): 8-angle guided rotation with segmentation capture
 *   Phase 3 (processing): Web Worker voxel carving + marching cubes
 */

import { useRef, useState, useCallback } from 'react';
import type { RecordFrame, BustAngleLabel, BustAngleCoverage, BustCaptureFrame } from '../types';
import { EMPTY_BUST_COVERAGE } from '../types';
import { estimatePose } from '../lib/poseDetect';
import { kabschProcrustes } from '../lib/poseEstimator';
import type { Mat3, Vec3 } from '../lib/poseEstimator';
import type { CarvingFrame } from '../lib/voxelCarver';
import type { WorkerOutput } from '../workers/bustWorker';
import * as THREE from 'three';

// ─── Constants ────────────────────────────────────────────────────────────────

const CAPTURE_INTERVAL_MS = 100;   // 10 fps during bust scan
const CONFIRM_FRAMES      = 12;    // frames to hold a bust angle (~1.2s)

// Target angles for bust scan (required + optional)
const REQUIRED_BUST_ANGLES: BustAngleLabel[] = ['front', 'frontLeft', 'left', 'right', 'frontRight'];
const ALL_BUST_ANGLES: BustAngleLabel[]       = [
  'front', 'frontLeft', 'left', 'backLeft',
  'back',  'backRight', 'right', 'frontRight',
  'up', 'down',
];

// Grid config (matches bustWorker)
const GRID_NX    = 96;
const GRID_NY    = 144;
const GRID_NZ    = 96;
const GRID_SCALE = 1.5;
const FOCAL_LEN  = 1.0;

// ─── Angle classification ─────────────────────────────────────────────────────

/**
 * Classify head yaw into one of the 8 horizontal bust angles.
 * yaw = lm[234].z - lm[454].z (positive = turning right).
 * We accumulate a running rotation angle by integrating yaw across frames.
 */
function classifyBustAngle(
  yaw: number,
  pitch: number,
  cumulativeYaw: number,
): BustAngleLabel {
  // Pitch overrides yaw for up/down
  if (pitch >  0.07) return 'up';
  if (pitch < -0.07) return 'down';

  // Normalise cumulative yaw to [-π, π]
  const angle = cumulativeYaw;
  const deg = angle * (180 / Math.PI);

  if (deg > -22.5  && deg <=  22.5) return 'front';
  if (deg >  22.5  && deg <=  67.5) return 'frontLeft';
  if (deg >  67.5  && deg <= 112.5) return 'left';
  if (deg > 112.5  || deg <= -112.5) return 'back';
  if (deg > -112.5 && deg <= -67.5) return 'backRight';
  if (deg > -67.5  && deg <= -22.5) return 'frontRight';
  if (deg > -67.5  && deg <= -22.5) return 'frontRight';
  return 'front';
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type BustPhase = 'idle' | 'face-scan' | 'bust-scan' | 'processing' | 'done';

export interface BustRecorderResult {
  phase:              BustPhase;
  faceCoverage:       { neutral: boolean; left: boolean; right: boolean };
  bustCoverage:       BustAngleCoverage;
  currentBustAngle:   BustAngleLabel | null;
  pendingProgress:    number;   // 0-1 for currently-confirming bust angle
  processingProgress: number;   // 0-1 during processing phase
  processingMessage:  string;
  faceLandmarks:      number[][] | null;
  bustGeometry:       THREE.BufferGeometry | null;
  start:              () => void;
  advanceToBustScan:  () => void;
  stop:               () => void;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useBustRecorder(
  captureFrame:    (() => RecordFrame | null) | null,
  captureMask:     (() => Promise<{ mask: Uint8Array; width: number; height: number } | null>) | null,
): BustRecorderResult {
  const [phase,              setPhase]              = useState<BustPhase>('idle');
  const [faceCoverage,       setFaceCoverage]       = useState({ neutral: false, left: false, right: false });
  const [bustCoverage,       setBustCoverage]       = useState<BustAngleCoverage>({ ...EMPTY_BUST_COVERAGE });
  const [currentBustAngle,   setCurrentBustAngle]   = useState<BustAngleLabel | null>(null);
  const [pendingProgress,    setPendingProgress]    = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingMessage,  setProcessingMessage]  = useState('');
  const [faceLandmarks,      setFaceLandmarks]      = useState<number[][] | null>(null);
  const [bustGeometry,       setBustGeometry]       = useState<THREE.BufferGeometry | null>(null);

  // Face scan state (mirrors useRecorder)
  const faceFramesRef     = useRef<Map<string, RecordFrame[]>>(new Map());
  const facePendingLabel  = useRef<string | null>(null);
  const facePendingBuf    = useRef<RecordFrame[]>([]);

  // Bust scan state
  const refLandmarksRef   = useRef<number[][] | null>(null);
  const bustFramesRef     = useRef<BustCaptureFrame[]>([]);
  const bustConfirmedRef  = useRef<Set<BustAngleLabel>>(new Set());
  const bustPendingLabel  = useRef<BustAngleLabel | null>(null);
  const bustPendingBuf    = useRef<RecordFrame[]>([]);
  const cumulativeYawRef  = useRef(0);
  const lastYawRef        = useRef(0);

  const intervalRef       = useRef<ReturnType<typeof setInterval> | null>(null);
  const workerRef         = useRef<Worker | null>(null);

  // ─── Helpers ───────────────────────────────────────────────────────────────

  function clearInterval_() {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  }

  function smoothLandmarks(frames: RecordFrame[]): number[][] {
    if (!frames.length) return [];
    const n = frames[0].landmarks.length;
    const sum = Array.from({ length: n }, () => [0,0,0]);
    for (const f of frames) for (let i = 0; i < n; i++) {
      sum[i][0] += f.landmarks[i][0];
      sum[i][1] += f.landmarks[i][1];
      sum[i][2] += f.landmarks[i][2];
    }
    return sum.map(v => [v[0]/frames.length, v[1]/frames.length, v[2]/frames.length]);
  }

  // ─── Phase 1: face scan ────────────────────────────────────────────────────

  function runFaceScanTick() {
    const frame = captureFrame?.();
    if (!frame) return;

    const pose = estimatePose(frame.landmarks);
    const label = pose.label;

    const confirmed = faceFramesRef.current;
    if (!confirmed.has(label)) {
      if (facePendingLabel.current === label) {
        facePendingBuf.current.push(frame);
      } else {
        facePendingLabel.current = label;
        facePendingBuf.current = [frame];
      }

      const pct = facePendingBuf.current.length / CONFIRM_FRAMES;
      if (label === 'neutral') setPendingProgress(Math.min(pct, 1));

      if (facePendingBuf.current.length >= CONFIRM_FRAMES) {
        confirmed.set(label, [...facePendingBuf.current]);
        facePendingLabel.current = null;
        facePendingBuf.current = [];
        setPendingProgress(0);

        const hasFace = { neutral: confirmed.has('neutral'), left: confirmed.has('left'), right: confirmed.has('right') };
        setFaceCoverage(hasFace);
      }
    }
  }

  // ─── Phase 2: bust scan ────────────────────────────────────────────────────

  async function runBustScanTick() {
    const frame = captureFrame?.();
    if (!frame) return;

    const pose = estimatePose(frame.landmarks);

    // Integrate yaw angle
    const dyaw = pose.yaw - lastYawRef.current;
    lastYawRef.current = pose.yaw;
    cumulativeYawRef.current += dyaw * 5; // scale factor for yaw units → radians approximation

    const angle = classifyBustAngle(pose.yaw, pose.pitch, cumulativeYawRef.current);
    setCurrentBustAngle(angle);

    const confirmed = bustConfirmedRef.current;
    if (!confirmed.has(angle)) {
      if (bustPendingLabel.current === angle) {
        bustPendingBuf.current.push(frame);
      } else {
        bustPendingLabel.current = angle;
        bustPendingBuf.current = [frame];
      }

      const pct = bustPendingBuf.current.length / CONFIRM_FRAMES;
      setPendingProgress(Math.min(pct, 1));

      if (bustPendingBuf.current.length >= CONFIRM_FRAMES) {
        // Capture segmentation mask for this confirmed angle
        const maskData = await captureMask?.();
        if (maskData && refLandmarksRef.current) {
          // Estimate camera pose from face landmarks
          const middleFrame = bustPendingBuf.current[Math.floor(bustPendingBuf.current.length / 2)];
          const poseResult = kabschProcrustes(refLandmarksRef.current, middleFrame.landmarks);

          if (poseResult.valid) {
            const bustFrame: BustCaptureFrame = {
              ...middleFrame,
              mask:  maskData.mask,
              maskW: maskData.width,
              maskH: maskData.height,
              R:     [...poseResult.R],
              t:     [...poseResult.t],
            };
            bustFramesRef.current.push(bustFrame);
          }
        }

        confirmed.add(angle);
        bustPendingLabel.current = null;
        bustPendingBuf.current = [];
        setPendingProgress(0);

        const newCov: BustAngleCoverage = { ...EMPTY_BUST_COVERAGE };
        for (const k of confirmed) newCov[k] = true;
        setBustCoverage(newCov);

        // Auto-advance to processing when all required angles done
        const allRequired = REQUIRED_BUST_ANGLES.every(a => confirmed.has(a));
        if (allRequired) {
          clearInterval_();
          runProcessing();
        }
      }
    } else {
      if (bustPendingLabel.current !== null) {
        bustPendingLabel.current = null;
        bustPendingBuf.current = [];
        setPendingProgress(0);
      }
    }
  }

  // ─── Phase 3: processing ───────────────────────────────────────────────────

  function runProcessing() {
    setPhase('processing');
    setProcessingProgress(0);
    setProcessingMessage('Initialising reconstruction…');

    // Terminate any previous worker
    workerRef.current?.terminate();

    const worker = new Worker(
      new URL('../workers/bustWorker.ts', import.meta.url),
      { type: 'module' },
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerOutput>) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        setProcessingProgress(msg.pct);
        setProcessingMessage(msg.message);
      } else if (msg.type === 'done') {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(msg.positions, 3));
        geo.setAttribute('normal',   new THREE.BufferAttribute(msg.normals,   3));
        geo.setIndex(new THREE.BufferAttribute(msg.indices, 1));
        setBustGeometry(geo);
        setPhase('done');
        setProcessingProgress(1);
        worker.terminate();
      } else if (msg.type === 'error') {
        console.error('Bust worker error:', msg.message);
        setPhase('done');
        worker.terminate();
      }
    };

    // Convert BustCaptureFrames to CarvingFrames
    const carvingFrames: CarvingFrame[] = bustFramesRef.current.map(bf => ({
      mask:   bf.mask,
      maskW:  bf.maskW,
      maskH:  bf.maskH,
      R:      bf.R as Mat3,
      t:      bf.t as Vec3,
      aspect: bf.maskH / bf.maskW,
    }));

    worker.postMessage({
      type: 'run',
      frames: carvingFrames,
      gridNx: GRID_NX,
      gridNy: GRID_NY,
      gridNz: GRID_NZ,
      gridScale: GRID_SCALE,
      focalLen: FOCAL_LEN,
    });
  }

  // ─── Public controls ───────────────────────────────────────────────────────

  const start = useCallback(() => {
    clearInterval_();
    faceFramesRef.current     = new Map();
    facePendingLabel.current  = null;
    facePendingBuf.current    = [];
    bustFramesRef.current     = [];
    bustConfirmedRef.current  = new Set();
    bustPendingLabel.current  = null;
    bustPendingBuf.current    = [];
    cumulativeYawRef.current  = 0;
    lastYawRef.current        = 0;

    setPhase('face-scan');
    setFaceCoverage({ neutral: false, left: false, right: false });
    setBustCoverage({ ...EMPTY_BUST_COVERAGE });
    setPendingProgress(0);
    setCurrentBustAngle(null);
    setFaceLandmarks(null);
    setBustGeometry(null);

    intervalRef.current = setInterval(runFaceScanTick, CAPTURE_INTERVAL_MS);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureFrame]);

  const advanceToBustScan = useCallback(() => {
    clearInterval_();

    // Capture reference landmarks from confirmed neutral face frames
    const neutralFrames = faceFramesRef.current.get('neutral') ?? [];
    const allFaceFrames = [...faceFramesRef.current.values()].flat();
    const refLms = smoothLandmarks(neutralFrames.length > 0 ? neutralFrames : allFaceFrames);
    refLandmarksRef.current = refLms;
    setFaceLandmarks(refLms);

    setPhase('bust-scan');
    lastYawRef.current = 0;
    cumulativeYawRef.current = 0;

    // Reset bust coverage except 'front' (auto-confirm from face scan's neutral)
    bustConfirmedRef.current = new Set<BustAngleLabel>(['front']);
    setBustCoverage({ ...EMPTY_BUST_COVERAGE, front: true });

    // Capture initial front mask immediately if possible
    if (captureMask && refLms.length > 0) {
      const frontFrame = neutralFrames[neutralFrames.length - 1];
      if (frontFrame) {
        captureMask().then(maskData => {
          if (maskData && refLandmarksRef.current) {
            const poseResult = kabschProcrustes(refLandmarksRef.current, frontFrame.landmarks);
            if (poseResult.valid) {
              bustFramesRef.current.push({
                ...frontFrame,
                mask:  maskData.mask,
                maskW: maskData.width,
                maskH: maskData.height,
                R:     [...poseResult.R],
                t:     [...poseResult.t],
              });
            }
          }
        });
      }
    }

    intervalRef.current = setInterval(() => { runBustScanTick(); }, CAPTURE_INTERVAL_MS);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureFrame, captureMask]);

  const stop = useCallback(() => {
    clearInterval_();
    if (phase === 'bust-scan' && bustFramesRef.current.length > 0) {
      runProcessing();
    } else {
      setPhase('done');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  return {
    phase,
    faceCoverage,
    bustCoverage,
    currentBustAngle,
    pendingProgress,
    processingProgress,
    processingMessage,
    faceLandmarks,
    bustGeometry,
    start,
    advanceToBustScan,
    stop,
  };
}
