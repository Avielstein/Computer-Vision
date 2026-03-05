export type FacePart = 'leftEye' | 'rightEye' | 'leftBrow' | 'rightBrow' | 'nose' | 'lips' | 'skin';

export type QualityState = 'lost' | 'ok' | 'good';

export interface RecordFrame {
  landmarks: number[][];  // 468 × [x, y, z]
  timestamp: number;
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
