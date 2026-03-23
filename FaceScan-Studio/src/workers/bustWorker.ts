/// <reference lib="webworker" />
/**
 * Web Worker: depth inference + TSDF fusion + marching cubes + vertex color baking.
 *
 * Pipeline:
 *   1. If colorBitmaps present: load Depth-Anything, run per-frame depth → TSDF fusion
 *      Also cache ImageData objects for vertex color baking in step 4.
 *   2. Else: fall back to voxel carving (visual hull).
 *   3. Marching cubes on the scalar field → geometry buffers.
 *   4. Vertex color baking: for each mesh vertex, find the best-facing captured frame
 *      and sample its color. Returns colors: Float32Array (R,G,B per vertex in [0,1]).
 */

import { carveAllViews, type CarvingFrame } from '../lib/voxelCarver';
import { marchingCubes } from '../lib/marchingCubes';
import { TSDFVolume, type TSDFFrame } from '../lib/tsdfFusion';

// ─── Message protocol ─────────────────────────────────────────────────────────

export type WorkerInput = {
  type:                 'run';
  frames:               CarvingFrame[];
  /** Subset of frames that have color bitmaps — parallel to colorBitmaps/frameLandmarks */
  colorFrames?:         CarvingFrame[];
  colorBitmaps?:        ImageBitmap[];
  frameLandmarks?:      number[][][];
  /** Per-color-frame frontalness 0–1. Higher = more frontal = prefer for depth inference. */
  frontalScores?:       number[];
  /** Depth scale calibrated from frontal frame; reused for frames where estimation fails. */
  calibratedDepthScale?: number;
  gridNx:               number;
  gridNy:               number;
  gridNz:               number;
  gridScale:            number;
  focalLen:             number;
};

export type WorkerOutput =
  | { type: 'progress'; message: string; pct: number }
  | { type: 'done'; positions: Float32Array; normals: Float32Array; indices: Uint32Array; colors: Float32Array }
  | { type: 'error'; message: string };

// ─── Depth model singleton ────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let depthPipeline: any = null;

async function getDepthPipeline(progress: (m: string, p: number) => void) {
  if (depthPipeline) return depthPipeline;
  progress('Loading depth model (first run: ~25 MB)…', 0.02);
  const { pipeline, env } = await import('@huggingface/transformers');
  env.allowLocalModels = false;
  env.useBrowserCache  = true;
  try {
    depthPipeline = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', { device: 'webgpu', dtype: 'fp16' });
    progress('Depth model ready (WebGPU)', 0.08);
  } catch {
    depthPipeline = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', { device: 'wasm', dtype: 'int8' });
    progress('Depth model ready (WASM)', 0.08);
  }
  return depthPipeline;
}

// ─── Bitmap → ImageData + depth ──────────────────────────────────────────────

async function bitmapToImageData(bitmap: ImageBitmap): Promise<ImageData> {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx    = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function inferDepth(pipe: any, imgData: ImageData): Promise<{ depthData: Float32Array; depthW: number; depthH: number }> {
  const result = await pipe(imgData);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tensor: any = result.predicted_depth;
  const rawData: Float32Array = tensor.data instanceof Float32Array ? tensor.data : new Float32Array(tensor.data);
  const [h, w] = tensor.dims.slice(-2);
  return { depthData: rawData, depthW: w, depthH: h };
}

// ─── Vertex color baking ─────────────────────────────────────────────────────

/**
 * For each mesh vertex, find the captured frame that faces it most directly,
 * project the vertex into that frame, and sample the color image.
 * Returns a Float32Array of R,G,B values per vertex, in [0,1].
 */
function bakeVertexColors(
  positions:  Float32Array,
  normals:    Float32Array,
  frames:     CarvingFrame[],
  imageDatas: ImageData[],
  focalLen:   number,
): Float32Array {
  const nVerts = positions.length / 3;
  const colors = new Float32Array(nVerts * 3).fill(0.78); // default warm grey

  for (let vi = 0; vi < nVerts; vi++) {
    const vx = positions[vi*3], vy = positions[vi*3+1], vz = positions[vi*3+2];
    const nx = normals[vi*3],   ny = normals[vi*3+1],   nz = normals[vi*3+2];

    let bestScore = 0.05; // minimum face-angle threshold
    let bestFi    = -1;
    let bestPx    = 0;
    let bestPy    = 0;
    let bestMaskW = 1;
    let bestMaskH = 1;

    for (let fi = 0; fi < frames.length; fi++) {
      const { R, t, maskW, maskH, mask, aspect } = frames[fi];

      // Project vertex to camera space
      const pcx = R[0]*vx + R[1]*vy + R[2]*vz + t[0];
      const pcy = R[3]*vx + R[4]*vy + R[5]*vz + t[1];
      const pcz = R[6]*vx + R[7]*vy + R[8]*vz + t[2];
      if (pcz >= 0) continue;

      const u  = (pcx / -pcz) * focalLen;
      const v  = (pcy / -pcz) * focalLen * aspect;
      const px = Math.round((u + 0.5) * maskW);
      const py = Math.round((v + 0.5) * maskH);
      if (px < 0 || px >= maskW || py < 0 || py >= maskH) continue;
      if (mask[py * maskW + px] === 0) continue; // background

      // Score = how directly this frame faces the vertex normal
      const ncz   = R[6]*nx + R[7]*ny + R[8]*nz;
      const score = -ncz; // higher = surface more frontal to this camera
      if (score > bestScore) {
        bestScore = score;
        bestFi    = fi;
        bestPx    = px;
        bestPy    = py;
        bestMaskW = maskW;
        bestMaskH = maskH;
      }
    }

    if (bestFi < 0) continue;

    // Scale from mask coords to color image coords
    const imgData = imageDatas[bestFi];
    const ipx = Math.min(imgData.width  - 1, Math.round(bestPx / bestMaskW * imgData.width));
    const ipy = Math.min(imgData.height - 1, Math.round(bestPy / bestMaskH * imgData.height));
    const i4  = (ipy * imgData.width + ipx) * 4;
    colors[vi*3]   = imgData.data[i4]   / 255;
    colors[vi*3+1] = imgData.data[i4+1] / 255;
    colors[vi*3+2] = imgData.data[i4+2] / 255;
  }

  return colors;
}

// ─── Worker handler ───────────────────────────────────────────────────────────

self.onmessage = async (e: MessageEvent<WorkerInput>) => {
  const msg = e.data;
  if (msg.type !== 'run') return;

  const progress = (message: string, pct: number) => {
    self.postMessage({ type: 'progress', message, pct } satisfies WorkerOutput);
  };

  try {
    const {
      frames, colorFrames, colorBitmaps, frameLandmarks,
      frontalScores, calibratedDepthScale,
      gridNx, gridNy, gridNz, gridScale, focalLen,
    } = msg;
    // colorFrames is the subset of frames that have bitmaps; falls back to frames if absent
    const bitmapFrames = colorFrames ?? frames;

    // Select at most 20 frames for depth inference, preferring frontal ones
    const MAX_DEPTH = 20;
    let depthIndices = bitmapFrames.map((_, i) => i);
    if (frontalScores && frontalScores.length === bitmapFrames.length) {
      depthIndices.sort((a, b) => (frontalScores[b] ?? 0) - (frontalScores[a] ?? 0));
    }
    depthIndices = depthIndices.slice(0, MAX_DEPTH).sort((a, b) => a - b);

    const voxelSizeX = (2 * gridScale) / gridNx;
    const voxelSizeY = (2 * gridScale) / gridNy;
    const voxelSizeZ = (2 * gridScale) / gridNz;

    let field:      Float32Array;
    let imageDatas: ImageData[] | null = null;

    // ── Depth-Anything + TSDF path ───────────────────────────────────────────
    if (colorBitmaps && colorBitmaps.length > 0 && colorBitmaps.length === bitmapFrames.length && frameLandmarks) {
      progress('Starting depth reconstruction…', 0.01);
      const pipe = await getDepthPipeline(progress);
      const tsdf = new TSDFVolume(gridNx, gridNy, gridNz, gridScale);

      // Rasterize all bitmap frames for color baking
      imageDatas = await Promise.all(colorBitmaps.map(bitmapToImageData));

      // Depth-infer only the selected subset (capped, frontal-preferred)
      const n = depthIndices.length;
      let globalDepthScale = calibratedDepthScale ?? null;

      for (let ii = 0; ii < n; ii++) {
        const i = depthIndices[ii];
        progress(`Depth inference ${ii + 1}/${n}…`, 0.1 + 0.45 * (ii / n));
        const imgData = imageDatas[i];

        const { depthData, depthW, depthH } = await inferDepth(pipe, imgData);

        const tsdfFrame: TSDFFrame = {
          depth:     depthData,
          depthW,
          depthH,
          mask:      bitmapFrames[i].mask,
          maskW:     bitmapFrames[i].maskW,
          maskH:     bitmapFrames[i].maskH,
          R:         bitmapFrames[i].R,
          t:         bitmapFrames[i].t,
          aspect:    bitmapFrames[i].aspect,
          landmarks: frameLandmarks[i],
        };

        // Try per-frame scale; fall back to calibrated scale
        let depthScale = TSDFVolume.estimateDepthScale(tsdfFrame, focalLen);
        if ((depthScale == null || depthScale <= 0) && globalDepthScale && globalDepthScale > 0) {
          depthScale = globalDepthScale;
        }
        if (depthScale == null || depthScale <= 0) {
          progress(`Frame ${i+1}: no depth scale, skipping`, 0.1 + 0.45 * (ii / n));
          continue;
        }
        // Store first successful scale as fallback for subsequent frames
        if (!globalDepthScale) globalDepthScale = depthScale;

        progress(`Integrating frame ${ii+1}/${n}…`, 0.55 + 0.3 * (ii / n));
        tsdf.integrate(tsdfFrame, focalLen, depthScale);
      }

      progress('Extracting surface…', 0.88);
      field = tsdf.toField();

    // ── Voxel carving fallback ────────────────────────────────────────────────
    } else {
      if (colorBitmaps && colorBitmaps.length > 0) {
        progress('Rasterising color frames…', 0.02);
        imageDatas = await Promise.all(colorBitmaps.map(bitmapToImageData));
      }
      progress(`Voxel carving (${frames.length} frames)…`, 0.05);
      const { field: carved } = carveAllViews(frames, gridNx, gridNy, gridNz, gridScale, focalLen, progress);
      field = carved;
    }

    // ── Marching cubes ────────────────────────────────────────────────────────
    progress('Marching cubes…', 0.91);
    const { positions, normals, indices } = marchingCubes(
      field, gridNx, gridNy, gridNz, 0.5,
      [voxelSizeX, voxelSizeY, voxelSizeZ],
      [-gridScale, -gridScale, -gridScale],
    );

    // ── Vertex color baking ──────────────────────────────────────────────────
    progress('Baking vertex colors…', 0.95);
    const colors = (imageDatas && imageDatas.length > 0 && imageDatas.length === bitmapFrames.length)
      ? bakeVertexColors(positions, normals, bitmapFrames, imageDatas, focalLen)
      : new Float32Array(positions.length).fill(0.78);

    progress('Done!', 1);

    self.postMessage(
      { type: 'done', positions, normals, indices, colors } satisfies WorkerOutput,
      [positions.buffer, normals.buffer, indices.buffer, colors.buffer],
    );
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) } satisfies WorkerOutput);
  }
};
