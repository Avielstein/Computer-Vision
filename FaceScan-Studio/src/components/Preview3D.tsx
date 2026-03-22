import { useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { buildFaceMesh } from '../lib/meshBuilder';
import { hullMaterial } from '../lib/bustMeshMerger';
import type { PartVisibility } from '../types';
import { PART_INDICES } from '../lib/faceParts';

const PART_KEYS = Object.keys(PART_INDICES) as (keyof typeof PART_INDICES)[];

interface FaceMeshObjectProps {
  landmarks: number[][];
  partVisibility: PartVisibility;
}

function FaceMeshObject({ landmarks, partVisibility }: FaceMeshObjectProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { geometry, materials, partNames } = useMemo(() => {
    return buildFaceMesh(landmarks);
  }, [landmarks]);

  // Apply visibility per part
  useEffect(() => {
    materials.forEach((mat, i) => {
      const name = partNames[i] as keyof PartVisibility;
      mat.visible = partVisibility[name] ?? true;
    });
  }, [partVisibility, materials, partNames]);

  return <mesh ref={meshRef} geometry={geometry} material={materials} />;
}

interface BustSceneProps {
  landmarks: number[][];
  partVisibility: PartVisibility;
  bustGeometry: THREE.BufferGeometry;
}

function BustScene({ landmarks, partVisibility, bustGeometry }: BustSceneProps) {
  const groupRef = useRef<THREE.Group>(null);
  const hullMat = useMemo(() => hullMaterial(), []);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.3;
    }
  });

  return (
    <group ref={groupRef}>
      <FaceMeshObject landmarks={landmarks} partVisibility={partVisibility} />
      <mesh geometry={bustGeometry} material={hullMat} />
    </group>
  );
}

interface FaceOnlySceneProps {
  landmarks: number[][];
  partVisibility: PartVisibility;
}

function FaceOnlyScene({ landmarks, partVisibility }: FaceOnlySceneProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { geometry, materials, partNames } = useMemo(() => {
    return buildFaceMesh(landmarks);
  }, [landmarks]);

  useEffect(() => {
    materials.forEach((mat, i) => {
      const name = partNames[i] as keyof PartVisibility;
      mat.visible = partVisibility[name] ?? true;
    });
  }, [partVisibility, materials, partNames]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.3;
    }
  });

  return <mesh ref={meshRef} geometry={geometry} material={materials} />;
}

interface Props {
  landmarks: number[][] | null;
  partVisibility: PartVisibility;
  bustGeometry?: THREE.BufferGeometry | null;
}

export function Preview3D({ landmarks, partVisibility, bustGeometry }: Props) {
  if (!landmarks) {
    return (
      <div style={styles.placeholder}>
        <p style={{ color: '#555' }}>Scan result will appear here</p>
        <p style={{ color: '#333', fontSize: 12, marginTop: 6 }}>
          Record a scan to generate 3D preview
        </p>
      </div>
    );
  }

  return (
    <div style={styles.canvas}>
      <Canvas
        camera={{ position: [0, 0, 2.5], fov: 45 }}
        style={{ borderRadius: 12 }}
      >
        <ambientLight intensity={0.25} />
        <directionalLight position={[1.5, 2, 3]} intensity={1.4} />
        <directionalLight position={[-2, 0.5, 1]} intensity={0.6} />
        <directionalLight position={[0, -2, 1]} intensity={0.25} />
        {bustGeometry ? (
          <BustScene
            landmarks={landmarks}
            partVisibility={partVisibility}
            bustGeometry={bustGeometry}
          />
        ) : (
          <FaceOnlyScene landmarks={landmarks} partVisibility={partVisibility} />
        )}
        <OrbitControls enablePan={false} minDistance={1} maxDistance={5} />
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
