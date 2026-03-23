/**
 * Single-phase bust/object recorder.
 *
 * Improvements over v1:
 *  - 2D grid-based optical flow replaces 1D strip NCC (much more robust tracking)
 *  - Pitch tracking in motion-only mode (rotX applied alongside rotY)
 *  - Continuous frame capture every MIN_YAW_STEP_DEG (not just on angle confirmation)
 *  - Calibrated depth scale: estimated once on frontal frame, reused for all frames
 *  - ScanMode: 'head' (face-landmark guided) | 'object' (optical flow only)
 *  - frameCount state exposed for progress display
 *  - frontalScores forwarded to worker to prioritise depth inference budget
 */

import { useRef, useState, useCallback } from 'react';
import type React from 'react';
import type { RecordFrame, BustAngleLabel, BustAngleCoverage, BustCaptureFrame } from '../types';
import { EMPTY_BUST_COVERAGE } from '../types';
import { estimatePose } from '../lib/poseDetect';
import { kabschProcrustes } from '../lib/poseEstimator';
import type { Mat3, Vec3 } from '../lib/poseEstimator';
import type { CarvingFrame } from '../lib/voxelCarver';
import type { WorkerOutput } from '../workers/bustWorker';
import { captureFlowFrame, estimateFlow, type FlowFrame } from '../lib/opticalFlow';
import * as THREE from 'three';

// ─── Constants ────────────────────────────────────────────────────────────────

const CAPTURE_INTERVAL_MS  = 150;
const CALIB_FRAMES         = 8;
const CONFIRM_FRAMES       = 10;   // frames held at angle before confirming
const MIN_YAW_STEP_DEG     = 1.5;  // minimum yaw change between captures (°)
const MAX_TOTAL_FRAMES     = 120;  // hard cap to avoid OOM in worker
const MAX_DEPTH_FRAMES     = 20;   // max frames sent for depth inference

const GRID_NX    = 128;
const GRID_NY    = 192;
const GRID_NZ    = 128;
const GRID_SCALE = 1.5;
const FOCAL_LEN  = 1.0;
const MASK_SIZE  = 256; // matches useSegmentation MASK_SIZE

// Frontal anchor landmarks (nose, chin, ears, cheekbones, eyes, etc.)
const FRONTAL_ANCHORS = [1, 4, 10, 152, 234, 454, 33, 263, 2, 13];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function classifyBustAngle(_yaw: number, pitch: number, cumulativeYaw: number): BustAngleLabel {
  if (pitch >  0.07) return 'up';
  if (pitch < -0.07) return 'down';
  const deg = cumulativeYaw * (180 / Math.PI);
  if (deg > -22.5  && deg <=  22.5)  return 'front';
  if (deg >  22.5  && deg <=  67.5)  return 'frontLeft';
  if (deg >  67.5  && deg <= 112.5)  return 'left';
  if (deg > 112.5  || deg <= -112.5) return 'back';
  if (deg > -112.5 && deg <= -67.5)  return 'backRight';
  if (deg > -67.5  && deg <= -22.5)  return 'frontRight';
  return 'front';
}

function rotY(theta: number): Mat3 {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [c, 0, s,  0, 1, 0,  -s, 0, c];
}

function rotX(theta: number): Mat3 {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [1, 0, 0,  0, c, -s,  0, s, c];
}

function mat3Mul(A: Mat3, B: Mat3): Mat3 {
  return [
    A[0]*B[0]+A[1]*B[3]+A[2]*B[6], A[0]*B[1]+A[1]*B[4]+A[2]*B[7], A[0]*B[2]+A[1]*B[5]+A[2]*B[8],
    A[3]*B[0]+A[4]*B[3]+A[5]*B[6], A[3]*B[1]+A[4]*B[4]+A[5]*B[7], A[3]*B[2]+A[4]*B[5]+A[5]*B[8],
    A[6]*B[0]+A[7]*B[3]+A[8]*B[6], A[6]*B[1]+A[7]*B[4]+A[8]*B[7], A[6]*B[2]+A[7]*B[5]+A[8]*B[8],
  ];
}

function smoothLandmarks(frames: RecordFrame[]): number[][] {
  if (!frames.length) return [];
  const n = frames[0].landmarks.length;
  const sum = Array.from({ length: n }, () => [0, 0, 0]);
  for (const f of frames)
    for (let i = 0; i < n; i++) {
      sum[i][0] += f.landmarks[i][0];
      sum[i][1] += f.landmarks[i][1];
      sum[i][2] += f.landmarks[i][2];
    }
  return sum.map(v => [v[0] / frames.length, v[1] / frames.length, v[2] / frames.length]);
}

/**
 * Fraction of frontal anchor landmarks visible near the optical axis.
 * z near 0 = frontal; |z| > 0.08 = side view.
 * Returns 0–1.
 */
function computeFrontalScore(landmarks: number[][] | null): number {
  if (!landmarks || landmarks.length < 468) return 0;
  let visible = 0;
  for (const li of FRONTAL_ANCHORS) {
    if (Math.abs(landmarks[li]?.[2] ?? 1) < 0.08) visible++;
  }
  return visible / FRONTAL_ANCHORS.length;
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type BustPhase    = 'idle' | 'scanning' | 'processing' | 'done';
export type TrackingMode = 'face' | 'motion';
export type ScanMode     = 'head' | 'object';

export interface BustRecorderResult {
  phase:              BustPhase;
  scanMode:           ScanMode;
  calibrated:         boolean;
  calibrationPct:     number;
  bustCoverage:       BustAngleCoverage;
  currentBustAngle:   BustAngleLabel | null;
  pendingProgress:    number;
  processingProgress: number;
  processingMessage:  string;
  faceLandmarks:      number[][] | null;
  bustGeometry:       THREE.BufferGeometry | null;
  trackingMode:       TrackingMode;
  canProcess:         boolean;
  paused:             boolean;
  frameCount:         number;
  start:              () => void;
  pause:              () => void;
  resume:             () => void;
  process:            () => void;
  stop:               () => void;
  rescanAngle:        (angle: BustAngleLabel) => void;
  setScanMode:        (mode: ScanMode) => void;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useBustRecorder(
  captureFrame:       (() => RecordFrame | null) | null,
  captureMask:        (() => Promise<{ mask: Uint8Array; width: number; height: number } | null>) | null,
  captureColorFrame?: (() => Promise<ImageBitmap | null>) | null,
  videoRef?:          React.RefObject<HTMLVideoElement | null>,
): BustRecorderResult {

  const [phase,              setPhase]              = useState<BustPhase>('idle');
  const [scanMode,           setScanModeState]      = useState<ScanMode>('head');
  const [calibrated,         setCalibrated]         = useState(false);
  const [calibrationPct,     setCalibrationPct]     = useState(0);
  const [bustCoverage,       setBustCoverage]       = useState<BustAngleCoverage>({ ...EMPTY_BUST_COVERAGE });
  const [currentBustAngle,   setCurrentBustAngle]   = useState<BustAngleLabel | null>(null);
  const [pendingProgress,    setPendingProgress]    = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingMessage,  setProcessingMessage]  = useState('');
  const [faceLandmarks,      setFaceLandmarks]      = useState<number[][] | null>(null);
  const [bustGeometry,       setBustGeometry]       = useState<THREE.BufferGeometry | null>(null);
  const [trackingMode,       setTrackingMode]       = useState<TrackingMode>('face');
  const [canProcess,         setCanProcess]         = useState(false);
  const [paused,             setPaused]             = useState(false);
  const [frameCount,         setFrameCount]         = useState(0);

  // ── Core state refs ─────────────────────────────────────────────────────────
  const calibBufRef      = useRef<RecordFrame[]>([]);
  const refLandmarksRef  = useRef<number[][] | null>(null);
  const bustFramesRef    = useRef<BustCaptureFrame[]>([]);
  const frontalScoresRef = useRef<number[]>([]);
  const bustConfirmedRef = useRef<Set<BustAngleLabel>>(new Set());
  const bustPendingLabel = useRef<BustAngleLabel | null>(null);
  const bustPendingBuf   = useRef<RecordFrame[]>([]);
  const cumulativeYawRef = useRef(0);
  const lastYawRef       = useRef(0);
  const lastCapturedYawRef = useRef<number>(0);

  // ── Motion tracking refs ─────────────────────────────────────────────────────
  const motionCanvasRef    = useRef<OffscreenCanvas | null>(null);
  const prevFlowFrameRef   = useRef<FlowFrame | null>(null);
  const lastValidRRef      = useRef<Mat3 | null>(null);
  const lastValidTRef      = useRef<Vec3 | null>(null);
  const motionYawAccRef    = useRef(0);
  const motionPitchAccRef  = useRef(0);

  // ── Depth calibration ────────────────────────────────────────────────────────
  const calibDepthScaleRef = useRef<number | null>(null);

  // ── Scan mode ref (for use inside async ticks) ───────────────────────────────
  const scanModeRef = useRef<ScanMode>('head');

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const workerRef   = useRef<Worker | null>(null);

  // ─── Helpers ─────────────────────────────────────────────────────────────────

  function clearInterval_() {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  }

  function startInterval() {
    clearInterval_();
    intervalRef.current = setInterval(() => { runScanTick(); }, CAPTURE_INTERVAL_MS);
  }

  // ─── Processing ──────────────────────────────────────────────────────────────

  function runProcessing() {
    setPhase('processing');
    setProcessingProgress(0);
    setProcessingMessage('Initialising…');
    workerRef.current?.terminate();

    const worker = new Worker(new URL('../workers/bustWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerOutput>) => {
      const m = e.data;
      if (m.type === 'progress') {
        setProcessingProgress(m.pct);
        setProcessingMessage(m.message);
      } else if (m.type === 'done') {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
        geo.setAttribute('normal',   new THREE.BufferAttribute(m.normals,   3));
        geo.setAttribute('color',    new THREE.BufferAttribute(m.colors,    3));
        geo.setIndex(new THREE.BufferAttribute(m.indices, 1));
        setBustGeometry(geo);
        setPhase('done');
        setProcessingProgress(1);
        worker.terminate();
      } else if (m.type === 'error') {
        console.error('Bust worker error:', m.message);
        setPhase('done');
        worker.terminate();
      }
    };

    // Kabsch places the face centre at z≈0 in head-space, so voxels behind the
    // face (z > 0) always appear "behind the camera" (pc[2] >= 0) and are never
    // carved → slab / elongated shape.  Offsetting t[2] by −GRID_SCALE places
    // the face centre at depth GRID_SCALE so the full grid is in front of the
    // camera and both front AND back of the head get carved correctly.
    const fixT = (t: Vec3): Vec3 => [t[0], t[1], t[2] - GRID_SCALE];

    // All frames → geometry (voxel carving)
    const allCarvingFrames: CarvingFrame[] = bustFramesRef.current.map(bf => ({
      mask: bf.mask, maskW: bf.maskW, maskH: bf.maskH,
      R: bf.R as Mat3, t: fixT(bf.t as Vec3),
      aspect: bf.maskH / bf.maskW,
    }));

    // Frames with bitmaps → depth inference + color baking
    const colorEntries = bustFramesRef.current.map((bf, i) => ({ bf, i })).filter(e => e.bf.colorBitmap != null);
    const colorCarvingFrames: CarvingFrame[] = colorEntries.map(({ bf }) => ({
      mask: bf.mask, maskW: bf.maskW, maskH: bf.maskH,
      R: bf.R as Mat3, t: fixT(bf.t as Vec3),
      aspect: bf.maskH / bf.maskW,
    }));
    const colorBitmaps   = colorEntries.map(({ bf }) => bf.colorBitmap as ImageBitmap);
    const frameLandmarks = colorEntries.map(({ bf }) => bf.landmarks);
    const frontalScores  = colorEntries.map(({ i }) => frontalScoresRef.current[i] ?? 0);
    const hasColor       = colorBitmaps.length > 0;

    worker.postMessage(
      {
        type: 'run',
        frames:              allCarvingFrames,
        colorFrames:         hasColor ? colorCarvingFrames : undefined,
        colorBitmaps:        hasColor ? colorBitmaps       : undefined,
        frameLandmarks:      hasColor ? frameLandmarks     : undefined,
        frontalScores:       hasColor ? frontalScores      : undefined,
        calibratedDepthScale: calibDepthScaleRef.current ?? undefined,
        gridNx: GRID_NX, gridNy: GRID_NY, gridNz: GRID_NZ,
        gridScale: GRID_SCALE, focalLen: FOCAL_LEN,
      },
      hasColor ? colorBitmaps : [],
    );
  }

  // ─── Main scan tick ───────────────────────────────────────────────────────────

  async function runScanTick() {
    const frame  = captureFrame?.() ?? null;
    const video  = videoRef?.current ?? null;
    const isObj  = scanModeRef.current === 'object';

    // ── Object mode: skip face calibration, use optical flow only ────────────
    if (isObj && !refLandmarksRef.current) {
      refLandmarksRef.current = []; // sentinel: empty = object mode, calibrated
      setCalibrated(true);
      setCalibrationPct(1);
    }

    // ── Auto-calibration (head mode only) ────────────────────────────────────
    if (!refLandmarksRef.current) {
      if (!frame) { setCalibrationPct(0); return; }
      const pose = estimatePose(frame.landmarks);
      if (Math.abs(pose.yaw) < 0.04 && Math.abs(pose.pitch) < 0.04) {
        calibBufRef.current.push(frame);
      } else {
        calibBufRef.current = [];
      }
      setCalibrationPct(Math.min(calibBufRef.current.length / CALIB_FRAMES, 1));

      if (calibBufRef.current.length >= CALIB_FRAMES) {
        const refLms = smoothLandmarks(calibBufRef.current);
        refLandmarksRef.current = refLms;
        setFaceLandmarks(refLms);
        setCalibrated(true);
        cumulativeYawRef.current = 0;
        lastYawRef.current       = 0;

        bustConfirmedRef.current.add('front');
        setBustCoverage({ ...EMPTY_BUST_COVERAGE, front: true });

        const lastFrame = calibBufRef.current[calibBufRef.current.length - 1];
        const pr = kabschProcrustes(refLms, lastFrame.landmarks);
        if (pr.valid) {
          // Always initialise pose refs — don't gate on mask availability
          lastValidRRef.current      = pr.R;
          lastValidTRef.current      = pr.t;
          lastCapturedYawRef.current = 0;

          const [maskData, colorBitmap] = await Promise.all([
            captureMask?.() ?? Promise.resolve(null),
            captureColorFrame?.() ?? Promise.resolve(null),
          ]);
          // Fall back to all-foreground mask if segmentation not ready yet
          const effectiveMask = maskData ?? {
            mask: new Uint8Array(MASK_SIZE * MASK_SIZE).fill(255),
            width: MASK_SIZE, height: MASK_SIZE,
          };
          const score = computeFrontalScore(lastFrame.landmarks);
          bustFramesRef.current.push({
            ...lastFrame, angle: 'front',
            mask: effectiveMask.mask, maskW: effectiveMask.width, maskH: effectiveMask.height,
            R: [...pr.R], t: [...pr.t],
            colorBitmap: colorBitmap ?? undefined,
          });
          frontalScoresRef.current.push(score);
          setFrameCount(bustFramesRef.current.length);
          setCanProcess(true);
        }
      }
      return;
    }

    // ── Yaw / pose estimation ─────────────────────────────────────────────────
    let currentYaw = 0, currentPitch = 0;
    let poseR: Mat3 | null = null, poseT: Vec3 | null = null;

    if (frame && !isObj) {
      setTrackingMode('face');
      const pose = estimatePose(frame.landmarks);
      currentYaw   = pose.yaw;
      currentPitch = pose.pitch;
      const dyaw = pose.yaw - lastYawRef.current;
      lastYawRef.current       = pose.yaw;
      cumulativeYawRef.current += dyaw * 5;
      motionYawAccRef.current   = 0;
      motionPitchAccRef.current = 0;

      const refLms = refLandmarksRef.current;
      if (refLms && refLms.length > 0) {
        const pr = kabschProcrustes(refLms, frame.landmarks);
        if (pr.valid) {
          poseR = pr.R; poseT = pr.t;
          lastValidRRef.current = pr.R;
          lastValidTRef.current = pr.t;
        }
      }
      prevFlowFrameRef.current = null;

    } else if (video) {
      setTrackingMode('motion');
      if (!motionCanvasRef.current) motionCanvasRef.current = new OffscreenCanvas(1, 1);
      const curr = captureFlowFrame(video, motionCanvasRef.current);

      if (curr && prevFlowFrameRef.current) {
        const flow = estimateFlow(prevFlowFrameRef.current, curr);
        if (flow.confidence > 0.15) {
          motionYawAccRef.current   += flow.yawDelta;
          motionPitchAccRef.current += flow.pitchDelta;
          cumulativeYawRef.current  += flow.yawDelta;
        }
      }
      prevFlowFrameRef.current = curr ?? prevFlowFrameRef.current;

      if (lastValidRRef.current && lastValidTRef.current) {
        poseR = mat3Mul(
          mat3Mul(lastValidRRef.current, rotY(motionYawAccRef.current)),
          rotX(motionPitchAccRef.current),
        );
        poseT = lastValidTRef.current;
      }
      currentYaw   = cumulativeYawRef.current / 5;
      currentPitch = motionPitchAccRef.current;

    } else {
      return;
    }

    // ── Angle classification ───────────────────────────────────────────────────
    const angle = classifyBustAngle(currentYaw, currentPitch, cumulativeYawRef.current);
    setCurrentBustAngle(angle);

    // ── Angle confirmation for radar ───────────────────────────────────────────
    const confirmed = bustConfirmedRef.current;
    if (!confirmed.has(angle)) {
      const bufFrame = frame ?? (bustPendingBuf.current.length > 0
        ? bustPendingBuf.current[bustPendingBuf.current.length - 1] : null);

      if (bufFrame) {
        if (bustPendingLabel.current === angle) {
          bustPendingBuf.current.push(bufFrame);
        } else {
          bustPendingLabel.current = angle;
          bustPendingBuf.current   = [bufFrame];
        }
        setPendingProgress(Math.min(bustPendingBuf.current.length / CONFIRM_FRAMES, 1));

        if (bustPendingBuf.current.length >= CONFIRM_FRAMES) {
          confirmed.add(angle);
          bustPendingLabel.current = null;
          bustPendingBuf.current   = [];
          setPendingProgress(0);
          const cov: BustAngleCoverage = { ...EMPTY_BUST_COVERAGE };
          for (const k of confirmed) cov[k] = true;
          setBustCoverage(cov);
        }
      }
    } else {
      if (bustPendingLabel.current !== null) {
        bustPendingLabel.current = null;
        bustPendingBuf.current   = [];
        setPendingProgress(0);
      }
    }

    // ── Continuous frame capture ───────────────────────────────────────────────
    if (!poseR || !poseT) return;
    if (bustFramesRef.current.length >= MAX_TOTAL_FRAMES) return;

    const yawDiff = Math.abs(cumulativeYawRef.current - lastCapturedYawRef.current) * (180 / Math.PI);
    if (yawDiff < MIN_YAW_STEP_DEG) return; // haven't moved enough yet

    const frontalScore = frame ? computeFrontalScore(frame.landmarks) : 0;

    const [maskData, colorBitmap] = await Promise.all([
      captureMask?.() ?? Promise.resolve(null),
      captureColorFrame?.() ?? Promise.resolve(null),
    ]);

    // Fall back to all-foreground mask if segmentation not ready yet
    const effectiveMask = maskData ?? {
      mask: new Uint8Array(MASK_SIZE * MASK_SIZE).fill(255),
      width: MASK_SIZE, height: MASK_SIZE,
    };

    // Use actual frame landmarks if face detected; empty array for motion-only frames.
    // Never pass reference (front-facing) landmarks with a side-view pose — it confuses
    // depth scale estimation in the worker (front landmarks don't match side-view R/t).
    const lmSnapshot = frame?.landmarks ?? [];

    bustFramesRef.current.push({
      landmarks: lmSnapshot,
      timestamp: performance.now(),
      angle,
      mask: effectiveMask.mask, maskW: effectiveMask.width, maskH: effectiveMask.height,
      R: [...poseR], t: [...poseT],
      colorBitmap: colorBitmap ?? undefined,
    });
    frontalScoresRef.current.push(frontalScore);
    lastCapturedYawRef.current = cumulativeYawRef.current;
    setFrameCount(bustFramesRef.current.length);
    setCanProcess(true);
  }

  // ─── Public controls ──────────────────────────────────────────────────────────

  const start = useCallback(() => {
    clearInterval_();
    refLandmarksRef.current    = null;
    calibBufRef.current        = [];
    bustFramesRef.current      = [];
    frontalScoresRef.current   = [];
    bustConfirmedRef.current   = new Set();
    bustPendingLabel.current   = null;
    bustPendingBuf.current     = [];
    cumulativeYawRef.current   = 0;
    lastYawRef.current         = 0;
    lastCapturedYawRef.current = 0;
    prevFlowFrameRef.current   = null;
    lastValidRRef.current      = null;
    lastValidTRef.current      = null;
    motionYawAccRef.current    = 0;
    motionPitchAccRef.current  = 0;
    calibDepthScaleRef.current = null;

    setPhase('scanning');
    setCalibrated(false);
    setCalibrationPct(0);
    setBustCoverage({ ...EMPTY_BUST_COVERAGE });
    setPendingProgress(0);
    setCurrentBustAngle(null);
    setFaceLandmarks(null);
    setBustGeometry(null);
    setTrackingMode('face');
    setCanProcess(false);
    setPaused(false);
    setFrameCount(0);

    startInterval();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureFrame, captureMask, captureColorFrame]);

  const pause = useCallback(() => {
    clearInterval_();
    setPaused(true);
  }, []);

  const resume = useCallback(() => {
    setPaused(false);
    setPhase('scanning');
    startInterval();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureFrame, captureMask, captureColorFrame]);

  const process = useCallback(() => {
    clearInterval_();
    if (bustFramesRef.current.length > 0) runProcessing();
    else setPhase('done');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => {
    clearInterval_();
    setPaused(false);
    setPhase('idle');
  }, []);

  const rescanAngle = useCallback((angle: BustAngleLabel) => {
    bustConfirmedRef.current.delete(angle);
    const remaining = bustFramesRef.current.filter(f => f.angle !== angle);
    const keptIndices = bustFramesRef.current.map((f, i) => ({ f, i }))
      .filter(({ f }) => f.angle !== angle)
      .map(({ i }) => i);
    bustFramesRef.current = remaining;
    frontalScoresRef.current = keptIndices.map(i => frontalScoresRef.current[i] ?? 0);

    const cov: BustAngleCoverage = { ...EMPTY_BUST_COVERAGE };
    for (const k of bustConfirmedRef.current) cov[k] = true;
    setBustCoverage(cov);
    setBustGeometry(null);
    setFrameCount(remaining.length);
    setPaused(false);
    setPhase('scanning');
    startInterval();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureFrame, captureMask, captureColorFrame]);

  const setScanMode = useCallback((mode: ScanMode) => {
    scanModeRef.current = mode;
    setScanModeState(mode);
  }, []);

  const MAX_DEPTH_FRAMES_UNUSED = MAX_DEPTH_FRAMES; // keep for worker

  return {
    phase, scanMode, calibrated, calibrationPct,
    bustCoverage, currentBustAngle, pendingProgress,
    processingProgress, processingMessage,
    faceLandmarks, bustGeometry, trackingMode,
    canProcess, paused, frameCount,
    start, pause, resume, process, stop, rescanAngle, setScanMode,
    // suppress TS unused warning
    _maxDepth: MAX_DEPTH_FRAMES_UNUSED,
  } as BustRecorderResult & { _maxDepth: number };
}
