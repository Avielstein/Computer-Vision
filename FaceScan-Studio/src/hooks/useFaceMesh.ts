import { useEffect, useRef, useState, useCallback } from 'react';
import type { QualityState, RecordFrame } from '../types';
import { PART_INDICES, PART_COLORS } from '../lib/faceParts';

// MediaPipe tesselation triangles — imported at runtime from the CDN bundle.
// We load the face_mesh script dynamically to avoid Vite bundling issues.

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

export interface FaceMeshState {
  quality: QualityState;
  landmarks: number[][] | null;  // 468 × [x, y, z] normalized
  captureFrame: () => RecordFrame | null;
}

const MEDIAPIPE_FACE_MESH_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js';

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src;
    s.crossOrigin = 'anonymous';
    s.onload = () => resolve();
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

export function useFaceMesh(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  cameraReady: boolean
): FaceMeshState {
  const faceMeshRef = useRef<FaceMeshInstance | null>(null);
  const animRef = useRef<number>(0);
  const landmarksRef = useRef<number[][] | null>(null);
  const [quality, setQuality] = useState<QualityState>('lost');
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);

  const drawOverlay = useCallback(
    (lms: Array<{ x: number; y: number; z: number }>, canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width, h = canvas.height;

      // Draw skin base (filled oval)
      const skinPts = PART_INDICES.skin.map(i => [lms[i].x * w, lms[i].y * h]);
      ctx.beginPath();
      ctx.moveTo(skinPts[0][0], skinPts[0][1]);
      skinPts.forEach(([x, y]) => ctx.lineTo(x, y));
      ctx.closePath();
      ctx.fillStyle = PART_COLORS.skin;
      ctx.fill();

      // Draw each named part as a filled convex hull (simple polygon)
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
    },
    []
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
