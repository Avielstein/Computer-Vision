/**
 * MediaPipe Selfie Segmentation hook.
 * Loaded from CDN at runtime (same pattern as useFaceMesh.ts).
 * Returns a function that captures a segmentation mask from the current video frame.
 */

import { useEffect, useRef, useCallback } from 'react';

const SELFIE_SEG_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js';

declare global {
  interface Window {
    SelfieSegmentation: new (config: object) => SelfieSegInstance;
  }
}

interface SelfieSegInstance {
  setOptions: (opts: object) => void;
  onResults: (cb: (results: SelfieSegResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
}

interface SelfieSegResults {
  segmentationMask: HTMLCanvasElement | ImageBitmap;
}

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

export interface SegmentationState {
  ready: boolean;
  /** Capture a segmentation mask from the current video frame. Returns null if not ready. */
  captureMask: () => Promise<{ mask: Uint8Array; width: number; height: number } | null>;
}

const MASK_SIZE = 256; // internal resolution for segmentation masks

export function useSegmentation(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  cameraReady: boolean,
): SegmentationState {
  const segRef   = useRef<SelfieSegInstance | null>(null);
  const readyRef = useRef(false);

  useEffect(() => {
    if (!cameraReady) return;
    let destroyed = false;

    async function init() {
      await loadScript(SELFIE_SEG_URL);
      if (destroyed) return;

      const seg = new window.SelfieSegmentation({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
      });

      seg.setOptions({
        modelSelection: 1, // 1 = landscape/general, better for full body
      });

      // We don't need continuous results — we'll call send() on demand
      seg.onResults(() => {});

      segRef.current = seg;
      readyRef.current = true;
    }

    init().catch(console.error);

    return () => {
      destroyed = true;
      segRef.current?.close();
      segRef.current = null;
      readyRef.current = false;
    };
  }, [cameraReady]);

  const captureMask = useCallback(async () => {
    const seg = segRef.current;
    const video = videoRef.current;
    if (!seg || !video || !readyRef.current) return null;

    return new Promise<{ mask: Uint8Array; width: number; height: number } | null>((resolve) => {
      seg.onResults((results: SelfieSegResults) => {
        // Rasterise the segmentation mask to a Uint8Array via OffscreenCanvas
        try {
          const offscreen = new OffscreenCanvas(MASK_SIZE, MASK_SIZE);
          const ctx = offscreen.getContext('2d')!;
          ctx.drawImage(results.segmentationMask as CanvasImageSource, 0, 0, MASK_SIZE, MASK_SIZE);
          const imageData = ctx.getImageData(0, 0, MASK_SIZE, MASK_SIZE);
          // Use alpha channel: 255 = foreground (person), 0 = background
          const mask = new Uint8Array(MASK_SIZE * MASK_SIZE);
          for (let i = 0; i < mask.length; i++) {
            mask[i] = imageData.data[i * 4 + 3]; // alpha channel
          }
          resolve({ mask, width: MASK_SIZE, height: MASK_SIZE });
        } catch {
          resolve(null);
        }
      });

      seg.send({ image: video }).catch(() => resolve(null));
    });
  }, [videoRef]);

  return {
    ready: readyRef.current,
    captureMask,
  };
}
