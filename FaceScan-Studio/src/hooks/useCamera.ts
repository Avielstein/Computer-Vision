import { useEffect, useRef, useState } from 'react';
import type React from 'react';

export function useCamera(externalVideoRef?: React.RefObject<HTMLVideoElement | null>) {
  const ownRef = useRef<HTMLVideoElement>(null);
  const videoRef = (externalVideoRef as React.RefObject<HTMLVideoElement> | undefined) ?? ownRef;
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      } catch (err) {
        setError('Camera access denied. Please allow camera permissions and reload.');
        console.error(err);
      }
    }

    start();

    return () => {
      stream?.getTracks().forEach(t => t.stop());
    };
  }, []);

  return { videoRef, ready, error };
}
