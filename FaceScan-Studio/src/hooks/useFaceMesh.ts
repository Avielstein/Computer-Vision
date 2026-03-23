import { useEffect, useRef, useState, useCallback } from 'react';
import type { QualityState, RecordFrame } from '../types';
import { PART_INDICES, PART_COLORS } from '../lib/faceParts';

declare global {
  interface Window {
    FaceMesh: new (config: object) => FaceMeshInstance;
    FACEMESH_TESSELATION: [number, number][];
    FACEMESH_RIGHT_EYE: [number, number][];
    FACEMESH_LEFT_EYE: [number, number][];
    FACEMESH_FACE_OVAL: [number, number][];
    FACEMESH_LIPS: [number, number][];
  }
}

interface FaceMeshInstance {
  setOptions: (opts: object) => void;
  onResults: (cb: (results: FaceMeshResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
}

interface FaceMeshResults {
  multiFaceLandmarks?: Array<Array<{ x: number; y: number; z: number }>>;
}

export type OverlayMode = 'none' | 'regions' | 'confidence' | 'mesh';

export interface FaceMeshState {
  quality:      QualityState;
  landmarks:    number[][] | null;
  captureFrame: () => RecordFrame | null;
}

const MEDIAPIPE_FACE_MESH_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js';

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src; s.crossOrigin = 'anonymous';
    s.onload = () => resolve();
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

// ─── Draw helpers ─────────────────────────────────────────────────────────────

function drawRegions(
  ctx: CanvasRenderingContext2D,
  lms: Array<{ x: number; y: number; z: number }>,
  w: number, h: number,
) {
  // Skin base
  const skinPts = PART_INDICES.skin.map(i => [lms[i].x * w, lms[i].y * h]);
  ctx.beginPath();
  ctx.moveTo(skinPts[0][0], skinPts[0][1]);
  skinPts.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.closePath();
  ctx.fillStyle = PART_COLORS.skin;
  ctx.fill();

  // Named parts
  const parts = ['rightEye', 'leftEye', 'rightBrow', 'leftBrow', 'nose', 'lips'] as const;
  parts.forEach(part => {
    const pts = PART_INDICES[part].map(i => [lms[i].x * w, lms[i].y * h]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fillStyle = PART_COLORS[part];
    ctx.fill();
    ctx.strokeStyle = PART_COLORS[part].replace(/[\d.]+\)$/, '0.9)');
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function drawConfidence(
  ctx: CanvasRenderingContext2D,
  lms: Array<{ x: number; y: number; z: number }>,
  w: number, h: number,
) {
  // z ≈ 0 = frontal (green / high confidence); |z| > 0.08 = side (red / low)
  for (const lm of lms) {
    const px = lm.x * w, py = lm.y * h;
    const conf = Math.max(0, 1 - Math.abs(lm.z) / 0.08);
    const r = Math.round((1 - conf) * 255);
    const g = Math.round(conf * 200);
    const grad = ctx.createRadialGradient(px, py, 0, px, py, 5);
    grad.addColorStop(0, `rgba(${r},${g},60,0.55)`);
    grad.addColorStop(1, `rgba(${r},${g},60,0)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawMesh(
  ctx: CanvasRenderingContext2D,
  lms: Array<{ x: number; y: number; z: number }>,
  w: number, h: number,
) {
  const tess = window.FACEMESH_TESSELATION;
  if (tess) {
    ctx.strokeStyle = 'rgba(129,140,248,0.3)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (const [a, b] of tess) {
      const la = lms[a], lb = lms[b];
      if (!la || !lb) continue;
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
    }
    ctx.stroke();
  }
  const oval = window.FACEMESH_FACE_OVAL;
  if (oval) {
    ctx.strokeStyle = 'rgba(192,132,252,0.75)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const [a, b] of oval) {
      const la = lms[a], lb = lms[b];
      if (!la || !lb) continue;
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
    }
    ctx.stroke();
  }
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useFaceMesh(
  videoRef:    React.RefObject<HTMLVideoElement | null>,
  canvasRef:   React.RefObject<HTMLCanvasElement | null>,
  cameraReady: boolean,
  overlayMode: OverlayMode = 'regions',
): FaceMeshState {
  const faceMeshRef    = useRef<FaceMeshInstance | null>(null);
  const animRef        = useRef<number>(0);
  const landmarksRef   = useRef<number[][] | null>(null);
  const overlayModeRef = useRef<OverlayMode>(overlayMode);
  const [quality,   setQuality]   = useState<QualityState>('lost');
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);

  // Keep ref in sync so the onResults closure always sees the current mode
  useEffect(() => { overlayModeRef.current = overlayMode; }, [overlayMode]);

  const drawOverlay = useCallback(
    (lms: Array<{ x: number; y: number; z: number }>, canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width, h = canvas.height;
      const mode = overlayModeRef.current;
      if (mode === 'none') return;
      if (mode === 'regions')    drawRegions(ctx, lms, w, h);
      if (mode === 'confidence') drawConfidence(ctx, lms, w, h);
      if (mode === 'mesh')       drawMesh(ctx, lms, w, h);
    },
    [],
  );

  useEffect(() => {
    if (!cameraReady) return;
    let destroyed = false;

    async function init() {
      await loadScript(MEDIAPIPE_FACE_MESH_URL);

      const fm = new window.FaceMesh({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      fm.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      fm.onResults((results: FaceMeshResults) => {
        const canvas = canvasRef.current;
        if (!canvas || destroyed) return;

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const raw = results.multiFaceLandmarks[0];
          const lms = raw.map(p => [p.x, p.y, p.z]);
          landmarksRef.current = lms;
          setLandmarks(lms);
          setQuality('good');
          drawOverlay(raw, canvas);
        } else {
          landmarksRef.current = null;
          setLandmarks(null);
          setQuality('lost');
          const ctx = canvas.getContext('2d');
          ctx?.clearRect(0, 0, canvas.width, canvas.height);
        }
      });

      faceMeshRef.current = fm;

      async function loop() {
        if (destroyed) return;
        const video = videoRef.current;
        if (video && video.readyState >= 2) {
          await fm.send({ image: video });
        }
        animRef.current = requestAnimationFrame(loop);
      }
      loop();
    }

    init().catch(console.error);

    return () => {
      destroyed = true;
      cancelAnimationFrame(animRef.current);
      faceMeshRef.current?.close();
    };
  }, [cameraReady, videoRef, canvasRef, drawOverlay]);

  const captureFrame = useCallback((): RecordFrame | null => {
    if (!landmarksRef.current) return null;
    return { landmarks: landmarksRef.current, timestamp: performance.now() };
  }, []);

  return { quality, landmarks, captureFrame };
}
