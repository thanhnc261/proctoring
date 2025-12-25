/**
 * Video Utilities
 *
 * Helper functions for video capture and frame processing
 */

/**
 * Request camera access
 */
export async function requestCameraAccess(
  constraints?: MediaStreamConstraints
): Promise<MediaStream> {
  const defaultConstraints: MediaStreamConstraints = {
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: 'user',
    },
    audio: false,
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(
      constraints || defaultConstraints
    );
    return stream;
  } catch (error) {
    console.error('❌ Error accessing camera:', error);
    throw new Error('Failed to access camera');
  }
}

// Reusable canvas for frame capture (performance optimization)
let captureCanvas: HTMLCanvasElement | null = null;
let captureCtx: CanvasRenderingContext2D | null = null;

/**
 * Capture a frame from a video element with optimizations
 * - Reuses canvas element (saves 1-2ms per frame)
 * - Lower default quality (0.5 instead of 0.8, saves 50-60% bandwidth)
 * - Supports dynamic resolution scaling
 */
export function captureFrame(
  video: HTMLVideoElement,
  quality = 0.5,  // Optimized: Reduced from 0.8 to 0.5 for better performance
  resolutionScale = 1.0  // Allow dynamic resolution scaling (0.5 = half resolution)
): string | null {
  if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
    return null;
  }

  try {
    // Optimization: Reuse canvas instead of creating new one each time
    if (!captureCanvas) {
      captureCanvas = document.createElement('canvas');
      captureCtx = captureCanvas.getContext('2d');

      if (!captureCtx) {
        console.error('❌ Failed to get canvas context');
        return null;
      }
    }

    // Calculate scaled dimensions
    const targetWidth = Math.floor(video.videoWidth * resolutionScale);
    const targetHeight = Math.floor(video.videoHeight * resolutionScale);

    // Resize canvas if needed
    if (captureCanvas.width !== targetWidth || captureCanvas.height !== targetHeight) {
      captureCanvas.width = targetWidth;
      captureCanvas.height = targetHeight;
    }

    // Draw the current video frame (scaled if resolution < 1.0)
    captureCtx!.drawImage(video, 0, 0, targetWidth, targetHeight);

    // Convert to base64 JPEG with optimized quality
    const dataUrl = captureCanvas.toDataURL('image/jpeg', quality);

    // Extract base64 data (remove "data:image/jpeg;base64," prefix)
    const base64Data = dataUrl.split(',')[1];

    return base64Data;
  } catch (error) {
    console.error('❌ Error capturing frame:', error);
    return null;
  }
}

/**
 * Reset the reusable canvas (call when changing video sources)
 */
export function resetCaptureCanvas(): void {
  captureCanvas = null;
  captureCtx = null;
}

/**
 * Stop a media stream
 */
export function stopMediaStream(stream: MediaStream | null): void {
  if (!stream) return;

  stream.getTracks().forEach((track) => {
    track.stop();
  });
}

/**
 * Check if camera is available
 */
export async function isCameraAvailable(): Promise<boolean> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.some((device) => device.kind === 'videoinput');
  } catch (error) {
    console.error('❌ Error checking camera availability:', error);
    return false;
  }
}

/**
 * Get list of available cameras
 */
export async function getCameraList(): Promise<MediaDeviceInfo[]> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  } catch (error) {
    console.error('❌ Error getting camera list:', error);
    return [];
  }
}

/**
 * Calculate frame rate
 */
export class FrameRateCalculator {
  private timestamps: number[] = [];
  private maxSamples = 30;

  addFrame(): void {
    const now = performance.now();
    this.timestamps.push(now);

    // Keep only recent samples
    if (this.timestamps.length > this.maxSamples) {
      this.timestamps.shift();
    }
  }

  getFPS(): number {
    if (this.timestamps.length < 2) {
      return 0;
    }

    const first = this.timestamps[0];
    const last = this.timestamps[this.timestamps.length - 1];
    const elapsed = last - first;

    if (elapsed === 0) {
      return 0;
    }

    return ((this.timestamps.length - 1) / elapsed) * 1000;
  }

  reset(): void {
    this.timestamps = [];
  }
}

/**
 * Adaptive Frame Rate Controller
 * Dynamically adjusts frame rate based on backend processing speed and motion detection
 */
export class AdaptiveFrameRateController {
  private currentFPS: number;
  private readonly minFPS: number;
  private readonly maxFPS: number;
  private readonly targetFPS: number;
  private processingTimes: number[] = [];
  private readonly maxSamples = 10;

  constructor(targetFPS = 5, minFPS = 1, maxFPS = 10) {
    this.targetFPS = targetFPS;
    this.currentFPS = targetFPS;
    this.minFPS = minFPS;
    this.maxFPS = maxFPS;
  }

  /**
   * Update frame rate based on backend processing time and motion
   */
  updateFrameRate(processingTimeMs: number, hasMotion: boolean): number {
    // Track processing times
    this.processingTimes.push(processingTimeMs);
    if (this.processingTimes.length > this.maxSamples) {
      this.processingTimes.shift();
    }

    // Calculate average processing time
    const avgProcessingTime =
      this.processingTimes.reduce((sum, time) => sum + time, 0) /
      this.processingTimes.length;

    // Adjust FPS based on backend performance and motion
    if (hasMotion && avgProcessingTime < 50) {
      // Motion detected and backend is fast - increase FPS
      this.currentFPS = Math.min(this.currentFPS + 0.5, this.maxFPS);
    } else if (avgProcessingTime > 100) {
      // Backend is slow - decrease FPS
      this.currentFPS = Math.max(this.currentFPS - 0.5, this.minFPS);
    } else if (!hasMotion) {
      // No motion - reduce to minimum FPS to save bandwidth
      this.currentFPS = Math.max(this.currentFPS - 0.5, this.minFPS);
    }

    return Math.round(this.currentFPS);
  }

  /**
   * Get current interval in milliseconds
   */
  getInterval(): number {
    return 1000 / this.currentFPS;
  }

  /**
   * Get current FPS
   */
  getCurrentFPS(): number {
    return Math.round(this.currentFPS);
  }

  /**
   * Get resolution scale based on current performance
   * Returns lower resolution (0.7-1.0) when backend is slow
   */
  getResolutionScale(): number {
    if (this.processingTimes.length < 3) {
      return 1.0; // Full resolution initially
    }

    const avgProcessingTime =
      this.processingTimes.reduce((sum, time) => sum + time, 0) /
      this.processingTimes.length;

    // Scale down resolution when backend is slow
    if (avgProcessingTime > 150) {
      return 0.7; // 70% resolution when very slow
    } else if (avgProcessingTime > 100) {
      return 0.85; // 85% resolution when slow
    }

    return 1.0; // Full resolution when fast
  }

  /**
   * Reset controller state
   */
  reset(): void {
    this.currentFPS = this.targetFPS;
    this.processingTimes = [];
  }
}
