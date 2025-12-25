/**
 * VideoCapture Component
 *
 * Captures video from the user's webcam and sends frames to the backend for analysis
 */

import { useEffect, useRef, useState } from 'react';
import {
  requestCameraAccess,
  stopMediaStream,
  captureFrame,
  FrameRateCalculator,
  AdaptiveFrameRateController,
} from '../utils/videoUtils';
import type { WebSocketClient } from '../services/websocket';
import { useProctoringStore } from '../stores/proctoringStore';

interface VideoCaptureProps {
  wsClient: WebSocketClient | null;
  isActive: boolean;
  targetFPS?: number;
}

export function VideoCapture({
  wsClient,
  isActive,
  targetFPS = 5,
}: VideoCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const fpsCalculator = useRef(new FrameRateCalculator());
  const adaptiveController = useRef(new AdaptiveFrameRateController(targetFPS, 1, 10));

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [currentFPS, setCurrentFPS] = useState(0);
  const [framesSent, setFramesSent] = useState(0);
  const [videoResolution, setVideoResolution] = useState({ width: 0, height: 0 });
  const [showCanvas, setShowCanvas] = useState(true); // Toggle for canvas visibility

  // Initialize camera
  useEffect(() => {
    let mounted = true;

    async function initCamera() {
      try {
        setCameraError(null);
        const stream = await requestCameraAccess();

        if (!mounted) {
          stopMediaStream(stream);
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsReady(true);
          // Update resolution once video metadata loads
          videoRef.current.onloadedmetadata = () => {
            if (videoRef.current) {
              setVideoResolution({
                width: videoRef.current.videoWidth,
                height: videoRef.current.videoHeight
              });
            }
          };
        }
      } catch (error) {
        console.error('❌ Failed to access camera:', error);
        setCameraError(
          error instanceof Error
            ? error.message
            : 'Failed to access camera'
        );
      }
    }

    initCamera();

    return () => {
      mounted = false;
      if (streamRef.current) {
        stopMediaStream(streamRef.current);
        streamRef.current = null;
      }
      setIsReady(false);
    };
  }, []);

  // Handle frame capture and sending with adaptive frame rate
  useEffect(() => {
    console.log('📹 VideoCapture state:', {
      isActive,
      isReady,
      hasWsClient: wsClient !== null,
      targetFPS
    });

    if (!isActive || !isReady || !wsClient) {
      // Stop capturing if inactive, not ready, or no WebSocket client
      if (intervalRef.current !== null) {
        console.log('⏸️  Stopping frame capture');
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    // Get latest analysis for adaptive control
    const latestAnalysis = useProctoringStore.getState().latestAnalysis;

    // Start capturing frames with adaptive rate
    let currentIntervalMs = 1000 / targetFPS;
    console.log(`▶️  Starting adaptive frame capture (target: ${targetFPS} FPS)`);

    const captureAndSend = () => {
      if (!videoRef.current) {
        console.warn('⚠️ Video ref not available');
        return;
      }

      // Get adaptive resolution scale and quality
      const resolutionScale = adaptiveController.current.getResolutionScale();
      const quality = resolutionScale < 1.0 ? 0.4 : 0.5; // Lower quality for scaled frames

      // Capture frame with optimized settings
      const frameData = captureFrame(videoRef.current, quality, resolutionScale);

      if (!frameData) {
        console.warn('⚠️ Failed to capture frame');
        return;
      }

      if (wsClient) {
        wsClient.sendFrame(frameData);
        setFramesSent((prev) => prev + 1);

        // Update FPS calculation
        fpsCalculator.current.addFrame();
        const actualFPS = Math.round(fpsCalculator.current.getFPS());
        setCurrentFPS(actualFPS);

        // Update adaptive controller with processing time
        const processingTime = latestAnalysis?.metadata?.processing_time_ms || 0;
        const hasMotion = latestAnalysis?.gaze?.deviation || false;

        const newFPS = adaptiveController.current.updateFrameRate(processingTime, hasMotion);

        // Dynamically adjust interval if FPS changed significantly
        const newIntervalMs = adaptiveController.current.getInterval();
        if (Math.abs(newIntervalMs - currentIntervalMs) > 50) {
          currentIntervalMs = newIntervalMs;

          // Restart interval with new rate
          if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
          }

          intervalRef.current = window.setInterval(captureAndSend, currentIntervalMs);
          console.log(`🔄 Adjusted frame rate: ${newFPS} FPS (interval: ${currentIntervalMs}ms)`);
        }
      }
    };

    // Start initial interval
    intervalRef.current = window.setInterval(captureAndSend, currentIntervalMs);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isActive, isReady, wsClient, targetFPS]);

  // Reset stats when becoming inactive
  useEffect(() => {
    if (!isActive) {
      setFramesSent(0);
      setCurrentFPS(0);
      fpsCalculator.current.reset();
      adaptiveController.current.reset();
      // Reset smoothed boxes
      smoothedFaceBox.current = null;
      smoothedLeftEye.current = null;
      smoothedRightEye.current = null;
    }
  }, [isActive]);

  // Get latest analysis from store
  const latestAnalysis = useProctoringStore((state) => state.latestAnalysis);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Smooth bounding box interpolation (adaptive movement - slower than actual face/eye)
  const smoothedFaceBox = useRef<[number, number, number, number] | null>(null);
  const smoothedLeftEye = useRef<[number, number, number, number] | null>(null);
  const smoothedRightEye = useRef<[number, number, number, number] | null>(null);

  // Draw detection overlay with requestAnimationFrame throttling
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    console.log('🎨 VideoCapture overlay effect triggered:', {
      hasVideo: !!video,
      hasCanvas: !!canvas,
      hasAnalysis: !!latestAnalysis,
      face_detected: latestAnalysis?.gaze?.face_detected,
      face_box: latestAnalysis?.gaze?.face_box,
    });

    if (!video || !canvas || !latestAnalysis) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Cancel previous animation frame if it exists
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    // Helper to smoothly interpolate bounding boxes with adaptive responsiveness
    // Uses velocity-based smoothing: faster movement = less smoothing
    const smoothBox = (
      current: [number, number, number, number] | null,
      target: [number, number, number, number]
    ): [number, number, number, number] => {
      if (!current) return target;

      // Calculate distance moved (normalized)
      const dx = Math.abs(target[0] - current[0]);
      const dy = Math.abs(target[1] - current[1]);
      const dw = Math.abs(target[2] - current[2]);
      const dh = Math.abs(target[3] - current[3]);
      const maxMovement = Math.max(dx, dy, dw, dh);

      // Adaptive smoothing factor based on velocity
      // Fast movement (>8%): 20% smoothing (80% new position)
      // Medium movement (3-8%): 40% smoothing (60% new position)
      // Slow movement (<3%): 60% smoothing (40% new position)
      let smoothFactor: number;
      if (maxMovement > 0.08) {
        // Fast movement - respond quickly
        smoothFactor = 0.2;
      } else if (maxMovement > 0.03) {
        // Medium movement - balanced smoothing
        smoothFactor = 0.4;
      } else {
        // Slow/micro movement - heavy smoothing to reduce jitter
        smoothFactor = 0.6;
      }

      // Apply adaptive interpolation
      const alpha = 1 - smoothFactor; // Inverse for new position weight
      return [
        current[0] * smoothFactor + target[0] * alpha, // x
        current[1] * smoothFactor + target[1] * alpha, // y
        current[2] * smoothFactor + target[2] * alpha, // w
        current[3] * smoothFactor + target[3] * alpha, // h
      ];
    };

    // Use requestAnimationFrame for smooth rendering (throttles to ~60fps max)
    animationFrameRef.current = requestAnimationFrame(() => {
      // Skip rendering if canvas is hidden
      if (!showCanvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      // Match canvas resolution to video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const { gaze } = latestAnalysis;
      if (!gaze.face_detected) return;

      // Apply smooth interpolation to bounding boxes for adaptive movement
      if (gaze.face_box) {
        smoothedFaceBox.current = smoothBox(smoothedFaceBox.current, gaze.face_box);
      }
      if (gaze.left_eye) {
        smoothedLeftEye.current = smoothBox(smoothedLeftEye.current, gaze.left_eye);
      }
      if (gaze.right_eye) {
        smoothedRightEye.current = smoothBox(smoothedRightEye.current, gaze.right_eye);
      }

      // Helper to draw bounding box with corners (thin borders)
      const drawBox = (box: [number, number, number, number], color: string, lineWidth: number = 2) => {
        const [x, y, w, h] = box;
        const px = x * canvas.width;
        const py = y * canvas.height;
        const pw = w * canvas.width;
        const ph = h * canvas.height;

        // Main rectangle with glow (thin border)
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.strokeRect(px, py, pw, ph);

        // Corner markers (no glow, thin)
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.lineWidth = lineWidth;
        const cornerLen = Math.min(pw, ph) * 0.2;

        ctx.beginPath(); ctx.moveTo(px, py + cornerLen); ctx.lineTo(px, py); ctx.lineTo(px + cornerLen, py); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px + pw - cornerLen, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + cornerLen); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px + pw, py + ph - cornerLen); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw - cornerLen, py + ph); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px + cornerLen, py + ph); ctx.lineTo(px, py + ph); ctx.lineTo(px, py + ph - cornerLen); ctx.stroke();
      };

      // Helper to draw ellipse for eyes (smooth oval shape)
      const drawEllipse = (box: [number, number, number, number], color: string, lineWidth: number = 2) => {
        const [x, y, w, h] = box;
        const px = x * canvas.width;
        const py = y * canvas.height;
        const pw = w * canvas.width;
        const ph = h * canvas.height;

        // Calculate center and radii
        const centerX = px + pw / 2;
        const centerY = py + ph / 2;
        const radiusX = pw / 2;
        const radiusY = ph / 2;

        // Draw ellipse with glow
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.stroke();

        // Reset shadow
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      };

      // Draw face box (smoothed, thin border)
      if (smoothedFaceBox.current) {
        const color = gaze.deviation ? '#f43f5e' : '#10b981';
        drawBox(smoothedFaceBox.current, color, 2.5); // Thin: 2.5px

        // Label
        const [x, y] = smoothedFaceBox.current;
        ctx.fillStyle = color;
        ctx.font = 'bold 14px monospace';
        ctx.shadowBlur = 0;
        ctx.fillText(gaze.deviation ? '⚠ ALERT' : 'NORMAL', x * canvas.width, y * canvas.height - 10);
      }

      // Draw eye ellipses (smoothed, thin border)
      if (smoothedLeftEye.current) drawEllipse(smoothedLeftEye.current, '#3b82f6', 2); // Thin: 2px
      if (smoothedRightEye.current) drawEllipse(smoothedRightEye.current, '#3b82f6', 2); // Thin: 2px
    });

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [latestAnalysis, isActive, showCanvas]);

  if (cameraError) {
    return (
      <div className="bg-rose-50 border border-rose-200 rounded-2xl p-6 text-center shadow-sm" data-testid="camera-error">
        <div className="text-rose-700 text-lg font-semibold mb-2">Camera access error</div>
        <div className="text-rose-600 text-sm">{cameraError}</div>
        <div className="mt-4 text-xs text-rose-600">
          Grant camera permissions and refresh to re-arm the live feed.
        </div>
      </div>
    );
  }

  return (
    <div className="glass rounded-2xl overflow-hidden" data-testid="video-capture-container">
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/50">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <span className="text-lg">📹</span> Live Feed
          </h3>
        </div>
        <div className="flex items-center gap-2 text-xs font-bold">
          {/* Canvas Overlay Toggle */}
          <button
            onClick={() => setShowCanvas(!showCanvas)}
            className={`px-2 py-1 rounded-md border transition-all ${
              showCanvas
                ? 'bg-blue-100 text-blue-700 border-blue-200 hover:bg-blue-200'
                : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
            }`}
            title={showCanvas ? 'Hide detection overlay' : 'Show detection overlay'}
          >
            {showCanvas ? '👁️ OVERLAY' : '👁️‍🗨️ OVERLAY'}
          </button>

          <div className="px-2 py-1 rounded-md bg-white/50 border border-white/60 text-slate-600">
            {targetFPS} FPS TARGET
          </div>
          <div
            className={`px-2 py-1 rounded-md border flex items-center gap-1.5 ${isActive
              ? 'bg-emerald-100 text-emerald-700 border-emerald-200'
              : 'bg-slate-100 text-slate-600 border-slate-200'
              }`}
            data-testid="monitoring-status-badge"
          >
            <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-emerald-600 animate-pulse' : 'bg-slate-400'}`} />
            {isActive ? 'LIVE' : 'PAUSED'}
          </div>
        </div>
      </div>

      {/* Video Feed */}
      <div className="relative bg-black aspect-video overflow-hidden group">

        {/* Placeholder / Empty State styling */}
        <div className="absolute inset-0 z-0">
          {!isReady && (
            <div className="w-full h-full flex flex-col items-center justify-center text-white/50 space-y-3">
              <div className="w-12 h-12 border-2 border-white/20 border-t-white/80 rounded-full animate-spin" />
              <p className="text-sm font-medium tracking-wide">Initializing Camera...</p>
            </div>
          )}
        </div>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          data-testid="video-element"
          className={`w-full h-full object-cover relative z-10 transition-opacity duration-500 ${isReady ? 'opacity-100' : 'opacity-0'}`}
          onLoadedMetadata={() => setIsReady(true)}
        />

        {/* Detection Overlay Canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 z-20 pointer-events-none w-full h-full"
        />

        {/* Overlay Grid (Cyberpunk-ish) */}
        <div className="absolute inset-0 z-20 pointer-events-none opacity-20 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:50px_50px]" />

        {/* Recording Indicator */}
        {isActive && (
          <div className="absolute inset-0 z-20 pointer-events-none border-[3px] border-rose-500/30 animate-pulse" />
        )}

        {/* Floating Stats */}
        <div className="absolute bottom-4 left-4 right-4 z-30 flex items-end justify-between opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <div className="bg-black/60 backdrop-blur-md rounded-lg p-2 text-white/90 text-xs font-mono space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-white/50">RES:</span>
              <span>{videoResolution.width}x{videoResolution.height}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-white/50">FPS:</span>
              <span data-testid="current-fps">{currentFPS}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-white/50">Sent:</span>
              <span data-testid="frames-sent">{framesSent}</span>
            </div>
          </div>

          {isActive && (
            <div className="bg-rose-500/90 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-bold animate-pulse flex items-center gap-2">
              <span className="w-2 h-2 bg-white rounded-full" />
              REC
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

