"""
Gaze Detection Module - Stream 1

This module determines where the candidate is looking by analyzing head pose
and detecting sustained deviations from the screen.

Technology: MediaPipe Face Mesh + Perspective-n-Point (PnP) algorithm
"""

import asyncio
import time
from collections import deque
from typing import Dict, Optional, Tuple, Deque

import cv2
import mediapipe as mp
import numpy as np

from app.config import settings


class GazeDetector:
    """
    Detects head pose and gaze direction using MediaPipe Face Mesh
    and PnP (Perspective-n-Point) algorithm.

    Attributes:
        mesh: MediaPipe Face Mesh solution
        model_points: 3D model points for PnP algorithm
        deviation_start_time: Timestamp when deviation started
        current_deviation_duration: Current duration of sustained deviation
    """

    def __init__(self):
        """Initialize MediaPipe Face Mesh and 3D model points."""
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 3D model points for solvePnP (approximate head model)
        # Points: nose tip, chin, left eye corner, right eye corner,
        #         left mouth corner, right mouth corner
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -63.6, -12.5),  # Chin
                (-43.3, 32.7, -26.0),  # Left eye left corner
                (43.3, 32.7, -26.0),  # Right eye right corner
                (-28.9, -28.9, -24.1),  # Left mouth corner
                (28.9, -28.9, -24.1),  # Right mouth corner
            ],
            dtype=np.float64,
        )

        # MediaPipe landmark indices for facial features
        self.idx_nose = 1
        self.idx_chin = 152
        self.idx_left_eye_outer = 33
        self.idx_right_eye_outer = 263
        self.idx_mouth_left = 61
        self.idx_mouth_right = 291

        # Eye indices for bounding box calculation
        # These cover the eye contour
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Deviation tracking
        self.deviation_start_time: Optional[float] = None
        self.current_deviation_duration: float = 0.0
        self.last_normal_time: Optional[float] = None  # Track when user last looked normal

        # Head pose smoothing (moving average)
        self.yaw_history: Deque[float] = deque(maxlen=settings.HEAD_POSE_SMOOTHING_WINDOW)
        self.pitch_history: Deque[float] = deque(maxlen=settings.HEAD_POSE_SMOOTHING_WINDOW)
        self.deviation_history: Deque[bool] = deque(maxlen=settings.HEAD_POSE_SMOOTHING_WINDOW)

        # Bounding box smoothing (5-frame moving average for stability)
        self.face_box_history: Deque[Tuple[float, float, float, float]] = deque(maxlen=settings.HEAD_POSE_SMOOTHING_WINDOW)
        self.left_eye_history: Deque[Tuple[float, float, float, float]] = deque(maxlen=settings.HEAD_POSE_SMOOTHING_WINDOW)
        self.right_eye_history: Deque[Tuple[float, float, float, float]] = deque(maxlen=settings.HEAD_POSE_SMOOTHING_WINDOW)

    async def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect head pose and gaze direction from a video frame.

        This method runs in an async context to allow concurrent processing
        but performs synchronous CV operations internally.

        Args:
            frame: Input video frame (BGR format from OpenCV)

        Returns:
            Dictionary containing:
                - face_detected (bool): Whether a face was detected
                - yaw (float): Horizontal head rotation in degrees
                - pitch (float): Vertical head rotation in degrees
                - roll (float): Head tilt in degrees
                - deviation (bool): Whether gaze is off-screen
                - deviation_duration (float): Duration of sustained deviation in seconds
                - confidence (float): Detection confidence score
                - landmarks (Optional[np.ndarray]): 2D facial landmark points
        """
        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._detect_sync, frame)
        return result

    def _detect_sync(self, frame: np.ndarray) -> Dict:
        """
        Synchronous detection method (runs in thread pool).

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Detection results dictionary
        """
        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Face Mesh
        results = self.mesh.process(rgb_frame)

        # No face detected
        if not results.multi_face_landmarks:
            # Reset deviation tracking completely
            self.deviation_start_time = None
            self.current_deviation_duration = 0.0
            self.last_normal_time = None

            return {
                "face_detected": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "deviation": False,
                "deviation_duration": 0.0,
                "confidence": 0.0,
                "landmarks_count": 0,
            }

        # Extract facial landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Get 2D image points from facial landmarks
        image_points = np.array(
            [
                (face_landmarks[self.idx_nose].x * w, face_landmarks[self.idx_nose].y * h),
                (face_landmarks[self.idx_chin].x * w, face_landmarks[self.idx_chin].y * h),
                (face_landmarks[self.idx_left_eye_outer].x * w, face_landmarks[self.idx_left_eye_outer].y * h),
                (face_landmarks[self.idx_right_eye_outer].x * w, face_landmarks[self.idx_right_eye_outer].y * h),
                (face_landmarks[self.idx_mouth_left].x * w, face_landmarks[self.idx_mouth_left].y * h),
                (face_landmarks[self.idx_mouth_right].x * w, face_landmarks[self.idx_mouth_right].y * h),
            ],
            dtype=np.float64,
        )

        # Calculate head pose using PnP
        yaw, pitch, roll = self._calculate_head_pose(image_points, w, h)

        # Add to history for smoothing
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

        # Calculate smoothed angles (moving average)
        smoothed_yaw = np.mean(list(self.yaw_history))
        smoothed_pitch = np.mean(list(self.pitch_history))

        # Check for attention deviation using smoothed angles
        minor_deviation, screen_deviation = self._check_deviation(smoothed_yaw, smoothed_pitch)

        # Track deviation consistency
        self.deviation_history.append(screen_deviation)

        # Only consider it a real deviation if it's consistent across the window
        deviation_consistency = sum(self.deviation_history) / len(self.deviation_history) if self.deviation_history else 0
        is_sustained_deviation = deviation_consistency >= settings.DEVIATION_CONSISTENCY_THRESHOLD

        # Update deviation duration with grace period logic
        self._update_deviation_duration_with_grace(is_sustained_deviation)

        # Get bounding boxes (raw)
        face_box_raw = self._get_face_bounding_box(face_landmarks)
        left_eye_box_raw = self._get_eye_bounding_box(face_landmarks, is_left=True)
        right_eye_box_raw = self._get_eye_bounding_box(face_landmarks, is_left=False)

        # Update history
        self.face_box_history.append(face_box_raw)
        self.left_eye_history.append(left_eye_box_raw)
        self.right_eye_history.append(right_eye_box_raw)

        # Calculate smoothed bounding boxes
        face_box = self._calculate_average_box(self.face_box_history)
        left_eye_box = self._calculate_average_box(self.left_eye_history)
        right_eye_box = self._calculate_average_box(self.right_eye_history)

        # Calculate confidence (placeholder - can be improved)
        confidence = 0.85

        return {
            "face_detected": True,
            "yaw": float(smoothed_yaw),  # Use smoothed values
            "pitch": float(smoothed_pitch),
            "roll": float(roll),
            "deviation": is_sustained_deviation,  # Only report sustained deviations
            "minor_deviation": minor_deviation,  # Track minor deviation (no alert)
            "deviation_duration": self.current_deviation_duration,
            "deviation_consistency": float(deviation_consistency),  # How consistent is the violation
            "confidence": confidence,
            "landmarks_count": len(image_points),
            "face_box": face_box,
            "left_eye": left_eye_box,
            "right_eye": right_eye_box,
        }

    def _calculate_head_pose(
        self, image_points: np.ndarray, w: int, h: int
    ) -> Tuple[float, float, float]:
        """
        Calculate head pose angles using solvePnP algorithm.

        Args:
            image_points: 2D facial landmark points
            w: Frame width
            h: Frame height

        Returns:
            Tuple of (yaw, pitch, roll) angles in degrees
        """
        # Camera matrix (assuming standard webcam with no distortion)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP to get rotation and translation vectors
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Calculate Euler angles from rotation matrix
        # Pitch: rotation around X-axis
        # Yaw: rotation around Y-axis
        # Roll: rotation around Z-axis
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        pitch = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        roll = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))

        return yaw, pitch, roll

    def _check_deviation(self, yaw: float, pitch: float) -> tuple[bool, bool]:
        """
        Check if head pose indicates attention deviation.

        Two-tier system:
        - Minor deviation (no alert): Slight looking away from camera
        - Screen deviation (HIGH alert): Looking at another screen/monitor

        Args:
            yaw: Horizontal head rotation in degrees
            pitch: Vertical head rotation in degrees

        Returns:
            Tuple of (minor_deviation, screen_deviation)
        """
        # Check for minor deviation (no alert)
        minor_yaw = abs(yaw) > settings.MINOR_YAW_THRESHOLD
        minor_pitch = abs(pitch) > settings.MINOR_PITCH_THRESHOLD
        minor_deviation = minor_yaw or minor_pitch

        # Check for screen deviation (HIGH alert - looking at another screen)
        screen_yaw = abs(yaw) > settings.SCREEN_YAW_THRESHOLD
        screen_pitch = abs(pitch) > settings.SCREEN_PITCH_THRESHOLD
        screen_deviation = screen_yaw or screen_pitch

        return minor_deviation, screen_deviation

    def _update_deviation_duration_with_grace(self, is_deviation: bool) -> None:
        """
        Update the deviation duration timer with grace period.

        The grace period prevents the system from resetting immediately when the user
        briefly looks back at the screen, reducing false negatives and alert cycling.

        Args:
            is_deviation: Whether current frame shows sustained deviation
        """
        current_time = time.time()

        if is_deviation:
            # Start tracking if not already started
            if self.deviation_start_time is None:
                self.deviation_start_time = current_time

            # Clear the "last normal" timestamp since we're deviating again
            self.last_normal_time = None

            # Calculate elapsed time
            self.current_deviation_duration = current_time - self.deviation_start_time
        else:
            # User is looking at screen normally

            # If we were tracking a deviation, start grace period
            if self.deviation_start_time is not None:
                # Mark when they started looking normal
                if self.last_normal_time is None:
                    self.last_normal_time = current_time

                # Check if grace period has elapsed
                time_since_normal = current_time - self.last_normal_time

                if time_since_normal >= settings.DEVIATION_GRACE_PERIOD:
                    # Grace period elapsed - fully reset
                    self.deviation_start_time = None
                    self.current_deviation_duration = 0.0
                    self.last_normal_time = None
                # else: Keep the deviation counter running during grace period
            else:
                # No active deviation, keep everything reset
                self.current_deviation_duration = 0.0
                self.last_normal_time = None

    def reset(self) -> None:
        """Reset the deviation tracking state and smoothing buffers."""
        self.deviation_start_time = None
        self.current_deviation_duration = 0.0
        self.last_normal_time = None
        self.yaw_history.clear()
        self.pitch_history.clear()
        self.deviation_history.clear()
        self.face_box_history.clear()
        self.left_eye_history.clear()
        self.right_eye_history.clear()



    def _get_face_bounding_box(self, landmarks) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of the face.

        Args:
            landmarks: MediaPipe normalized landmarks

        Returns:
            Tuple of (x, y, w, h) normalized coordinates
        """
        x_min = min([lm.x for lm in landmarks])
        y_min = min([lm.y for lm in landmarks])
        x_max = max([lm.x for lm in landmarks])
        y_max = max([lm.y for lm in landmarks])

        return x_min, y_min, x_max - x_min, y_max - y_min

    def _get_eye_bounding_box(self, landmarks, is_left: bool) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of an eye.

        Args:
            landmarks: MediaPipe normalized landmarks
            is_left: True for left eye, False for right eye

        Returns:
            Tuple of (x, y, w, h) normalized coordinates
        """
        indices = self.left_eye_indices if is_left else self.right_eye_indices
        eye_landmarks = [landmarks[i] for i in indices]

        x_min = min([lm.x for lm in eye_landmarks])
        y_min = min([lm.y for lm in eye_landmarks])
        x_max = max([lm.x for lm in eye_landmarks])
        y_max = max([lm.y for lm in eye_landmarks])

        # Add some padding
        padding_x = (x_max - x_min) * 0.2
        padding_y = (y_max - y_min) * 0.2

        return (
            max(0.0, x_min - padding_x),
            max(0.0, y_min - padding_y),
            (x_max - x_min) + 2 * padding_x,
            (y_max - y_min) + 2 * padding_y
        )

    def _calculate_average_box(
        self, history: Deque[Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the average bounding box from history.

        Args:
            history: Deque of (x, y, w, h) tuples

        Returns:
            Averaged (x, y, w, h) tuple
        """
        if not history:
            return 0.0, 0.0, 0.0, 0.0

        n = len(history)
        avg_x = sum(box[0] for box in history) / n
        avg_y = sum(box[1] for box in history) / n
        avg_w = sum(box[2] for box in history) / n
        avg_h = sum(box[3] for box in history) / n

        return avg_x, avg_y, avg_w, avg_h

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "mesh"):
            self.mesh.close()
