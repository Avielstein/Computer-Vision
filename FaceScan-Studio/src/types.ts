export type FacePart = 'leftEye' | 'rightEye' | 'leftBrow' | 'rightBrow' | 'nose' | 'lips' | 'skin';

export type QualityState = 'lost' | 'ok' | 'good';

export type PoseLabel = 'neutral' | 'left' | 'right' | 'up' | 'down';

export interface PoseCoverage {
  neutral: boolean;
  left: boolean;
  right: boolean;
  up: boolean;
  down: boolean;
}

export interface RecordFrame {
  landmarks: number[][];  // 468 × [x, y, z]
  timestamp: number;
}

// ─── Bust scan types ──────────────────────────────────────────────────────────

export type BustAngleLabel =
  | 'front' | 'frontLeft' | 'left' | 'backLeft'
  | 'back'  | 'backRight' | 'right' | 'frontRight'
  | 'up' | 'down';

export interface BustAngleCoverage {
  front: boolean; frontLeft: boolean; left: boolean; backLeft: boolean;
  back: boolean;  backRight: boolean; right: boolean; frontRight: boolean;
  up: boolean; down: boolean;
  [key: string]: boolean;
}

export const EMPTY_BUST_COVERAGE: BustAngleCoverage = {
  front: false, frontLeft: false, left: false, backLeft: false,
  back: false,  backRight: false, right: false, frontRight: false,
  up: false, down: false,
};

export interface BustCaptureFrame extends RecordFrame {
  mask:   Uint8Array;
  maskW:  number;
  maskH:  number;
  /** 3×3 rotation matrix (row-major): head-space → camera-space */
  R:      number[];
  /** 3-vector translation */
  t:      number[];
}

export interface PartVisibility {
  leftEye: boolean;
  rightEye: boolean;
  leftBrow: boolean;
  rightBrow: boolean;
  nose: boolean;
  lips: boolean;
  skin: boolean;
}

export const DEFAULT_PART_VISIBILITY: PartVisibility = {
  leftEye: true,
  rightEye: true,
  leftBrow: true,
  rightBrow: true,
  nose: true,
  lips: true,
  skin: true,
};
