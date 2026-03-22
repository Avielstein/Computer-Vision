/// <reference lib="webworker" />
/**
 * Web Worker: receives carving frames, runs voxel carving + marching cubes,
 * and sends geometry buffers back to the main thread as transferable objects.
 */

import { carveAllViews, type CarvingFrame } from '../lib/voxelCarver';
import { marchingCubes } from '../lib/marchingCubes';

// ─── Message protocol ─────────────────────────────────────────────────────────

export type WorkerInput =
  | {
      type: 'run';
      frames: CarvingFrame[];
      gridNx: number;
      gridNy: number;
      gridNz: number;
      gridScale: number;
      focalLen: number;
    };

export type WorkerOutput =
  | { type: 'progress'; message: string; pct: number }
  | { type: 'done'; positions: Float32Array; normals: Float32Array; indices: Uint32Array }
  | { type: 'error'; message: string };

// ─── Worker message handler ───────────────────────────────────────────────────

self.onmessage = (e: MessageEvent<WorkerInput>) => {
  const msg = e.data;

  if (msg.type === 'run') {
    try {
      const { frames, gridNx, gridNy, gridNz, gridScale, focalLen } = msg;

      const progress = (message: string, pct: number) => {
        self.postMessage({ type: 'progress', message, pct } satisfies WorkerOutput);
      };

      // Step 1: Voxel carving
      progress('Starting voxel carving…', 0);
      const { grid, field } = carveAllViews(
        frames, gridNx, gridNy, gridNz, gridScale, focalLen, progress,
      );

      // Step 2: Marching cubes
      progress('Running marching cubes…', 0.92);

      const voxelSizeX = (2 * gridScale) / gridNx;
      const voxelSizeY = (2 * gridScale) / gridNy;
      const voxelSizeZ = (2 * gridScale) / gridNz;

      const { positions, normals, indices } = marchingCubes(
        field,
        grid.Nx, grid.Ny, grid.Nz,
        0.5,
        [voxelSizeX, voxelSizeY, voxelSizeZ],
        [-gridScale, -gridScale, -gridScale],
      );

      progress('Done!', 1);

      // Transfer buffers (zero-copy)
      self.postMessage(
        {
          type: 'done',
          positions,
          normals,
          indices,
        } satisfies WorkerOutput,
        [positions.buffer, normals.buffer, indices.buffer],
      );
    } catch (err) {
      self.postMessage({
        type: 'error',
        message: String(err),
      } satisfies WorkerOutput);
    }
  }
};
