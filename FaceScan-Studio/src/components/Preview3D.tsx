import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { buildPrintableBust, CAM_ASPECT, PART_3D_COLORS } from '../lib/meshBuilder';
import { hullMaterial } from '../lib/bustMeshMerger';
import { PART_INDICES } from '../lib/faceParts';
import type { PartVisibility } from '../types';

export type RenderMode = 'mono' | 'depth' | 'texture' | 'segmented';

// ─── Per-mode material factories ──────────────────────────────────────────────

function monoMaterial() {
  return hullMaterial();
}

function textureMaterial() {
  return new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.7,
    metalness: 0.05,
  });
}

/** Map z-coords → heat-map vertex colors and return a vertexColors material. */
function buildDepthGeometry(source: THREE.BufferGeometry): THREE.BufferGeometry {
  const pos = source.attributes.position;
  if (!pos) return source;

  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i);
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  const range = zMax - zMin || 1;
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const t = (pos.getZ(i) - zMin) / range; // 0=back, 1=front
    // heat map: blue → cyan → green → yellow → red
    const r = Math.max(0, Math.min(1, 2 * t - 1));
    const g = Math.max(0, Math.min(1, t < 0.5 ? 2 * t : 2 - 2 * t));
    const b = Math.max(0, Math.min(1, 1 - 2 * t));
    colors[i * 3]     = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  const geo = source.clone();
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geo;
}

/**
 * Color vertices by nearest MediaPipe face region.
 * Transforms landmarks to Three.js world space (same as buildPrintableBust),
 * then KNN-assigns each vertex the color of its nearest landmark's part.
 */
function buildSegmentedGeometry(
  source:    THREE.BufferGeometry,
  landmarks: number[][] | null,
): THREE.BufferGeometry {
  const pos = source.attributes.position;
  if (!pos) return source;
  const nVerts = pos.count;
  const colors = new Float32Array(nVerts * 3);

  // Build per-landmark color array from PART_INDICES
  const partKeys = Object.keys(PART_INDICES) as (keyof typeof PART_INDICES)[];
  const skinColor = new THREE.Color(PART_3D_COLORS['skin'] ?? 0xc8a078);
  const lmColor: THREE.Color[] = Array.from({ length: 468 }, () => skinColor);
  for (const key of partKeys) {
    const c = new THREE.Color(PART_3D_COLORS[key] ?? 0xc8a078);
    for (const idx of PART_INDICES[key]) {
      lmColor[idx] = c;
    }
  }

  if (landmarks && landmarks.length >= 468) {
    // Transform landmarks to Three.js space — matches buildPrintableBust
    const lx = new Float32Array(468);
    const ly = new Float32Array(468);
    const lz = new Float32Array(468);
    for (let i = 0; i < 468; i++) {
      lx[i] =  (landmarks[i][0] - 0.5) * 2;
      ly[i] = -(landmarks[i][1] - 0.5) * 2 * CAM_ASPECT;
      lz[i] = -landmarks[i][2] * 8;
    }

    for (let vi = 0; vi < nVerts; vi++) {
      const vx = pos.getX(vi), vy = pos.getY(vi), vz = pos.getZ(vi);
      let best = 0, bestD = Infinity;
      for (let li = 0; li < 468; li++) {
        const dx = vx - lx[li], dy = vy - ly[li], dz = vz - lz[li];
        const d = dx*dx + dy*dy + dz*dz;
        if (d < bestD) { bestD = d; best = li; }
      }
      const c = lmColor[best];
      colors[vi * 3]     = c.r;
      colors[vi * 3 + 1] = c.g;
      colors[vi * 3 + 2] = c.b;
    }
  } else {
    // No landmarks yet — fall back to skin tone
    colors.fill(skinColor.r);
    for (let i = 0; i < nVerts; i++) {
      colors[i * 3]     = skinColor.r;
      colors[i * 3 + 1] = skinColor.g;
      colors[i * 3 + 2] = skinColor.b;
    }
  }

  const geo = source.clone();
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geo;
}

// ─── Spinning mesh ─────────────────────────────────────────────────────────────

function SpinningMesh({
  geometry,
  renderMode,
  landmarks,
}: {
  geometry:   THREE.BufferGeometry;
  renderMode: RenderMode;
  landmarks:  number[][] | null;
}) {
  const ref = useRef<THREE.Mesh>(null);

  const [displayGeo, mat] = useMemo(() => {
    switch (renderMode) {
      case 'texture': {
        const hasColor = !!geometry.attributes.color;
        return [geometry, hasColor ? textureMaterial() : monoMaterial()];
      }
      case 'depth': {
        const g = buildDepthGeometry(geometry);
        return [g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 })];
      }
      case 'segmented': {
        const g = buildSegmentedGeometry(geometry, landmarks);
        return [g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7, side: THREE.DoubleSide })];
      }
      case 'mono':
      default:
        return [geometry, monoMaterial()];
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, renderMode, landmarks]);

  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.35;
  });

  return <mesh ref={ref} geometry={displayGeo} material={mat} />;
}

// ─── Public component ──────────────────────────────────────────────────────────

interface Props {
  landmarks:       number[][] | null;
  partVisibility?: PartVisibility;
  bustGeometry?:   THREE.BufferGeometry | null;
  renderMode?:     RenderMode;
}

export function Preview3D({ landmarks, bustGeometry, renderMode = 'mono' }: Props) {
  const faceGeometry = useMemo(() => {
    if (!landmarks) return null;
    return buildPrintableBust(landmarks);
  }, [landmarks]);

  const activeGeometry = bustGeometry ?? faceGeometry;

  if (!activeGeometry) {
    return (
      <div style={styles.placeholder}>
        <p style={{ color: '#555' }}>Scan result will appear here</p>
        <p style={{ color: '#333', fontSize: 12, marginTop: 6 }}>
          Complete a scan to generate 3D preview
        </p>
      </div>
    );
  }

  return (
    <div style={styles.canvas}>
      <Canvas
        camera={{ position: [0, 0, 2.8], fov: 42 }}
        style={{ borderRadius: 12 }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[2, 3, 4]}   intensity={1.8} />
        <directionalLight position={[-2, 1, 2]}  intensity={0.7} />
        <directionalLight position={[0, -2, 1]}  intensity={0.3} />
        <SpinningMesh geometry={activeGeometry} renderMode={renderMode} landmarks={landmarks} />
        <OrbitControls enablePan={false} minDistance={1} maxDistance={6} />
      </Canvas>
      <p style={styles.hint}>Drag to rotate · Scroll to zoom</p>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  placeholder: {
    width: '100%',
    height: 480,
    borderRadius: 12,
    border: '1px dashed #2a2a3a',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#0d0d15',
  },
  canvas: {
    width: '100%',
    height: 480,
    borderRadius: 12,
    overflow: 'hidden',
    border: '1px solid #2a2a3a',
    background: '#0d0d15',
    position: 'relative',
  },
  hint: {
    position: 'absolute',
    bottom: 10,
    left: 0,
    right: 0,
    textAlign: 'center',
    fontSize: 11,
    color: '#333',
    pointerEvents: 'none',
  },
};
