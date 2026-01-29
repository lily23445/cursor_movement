import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
import threading
import time
import keyboard

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

mouse_control_enabled = True
filter_length = 8

mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

calibration_offset_yaw = 0
calibration_offset_pitch = 0

ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Key landmarks
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1
}

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

# mouse thread
def mouse_mover():
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)

threading.Thread(target=mouse_mover, daemon=True).start()

# main Loop 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera not opened")
        break

    cv2.imshow("showing", frame)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Extract key points
        pts = {k: landmark_to_np(face_landmarks[i], w, h) for k, i in LANDMARKS.items()}
        left, right, top, bottom, front = pts.values()

        # Head coordinate axes
        right_axis = right - left
        right_axis /= np.linalg.norm(right_axis)

        up_axis = top - bottom
        up_axis /= np.linalg.norm(up_axis)

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)
        forward_axis = -forward_axis  # face outward

        center = (left + right + top + bottom + front) / 5

        # Smoothing
        ray_origins.append(center)
        ray_directions.append(forward_axis)

        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)

        reference_forward = np.array([0, 0, -1])

        # yaw
        xz = np.array([avg_direction[0], 0, avg_direction[2]])
        xz /= np.linalg.norm(xz)
        yaw = math.degrees(math.acos(np.clip(np.dot(reference_forward, xz), -1, 1)))
        if avg_direction[0] < 0:
            yaw = -yaw

        # pitch
        yz = np.array([0, avg_direction[1], avg_direction[2]])
        yz /= np.linalg.norm(yz)
        pitch = math.degrees(math.acos(np.clip(np.dot(reference_forward, yz), -1, 1)))
        if avg_direction[1] > 0:
            pitch = -pitch

        # Normalize angles
        yaw = abs(yaw) if yaw < 0 else 360 - yaw
        pitch = 360 + pitch if pitch < 0 else pitch

        raw_yaw, raw_pitch = yaw, pitch

        # Calibration
        yaw += calibration_offset_yaw
        pitch += calibration_offset_pitch

  
        if abs(yaw - 180) < 2:
            yaw = 180
        if abs(pitch - 180) < 3:
            pitch = 180

        yaw_range = 20
        pitch_range = 10

        screen_x = int(((yaw - (180 - yaw_range)) / (2 * yaw_range)) * MONITOR_WIDTH)
        screen_y = int(((180 + pitch_range - pitch) / (2 * pitch_range)) * MONITOR_HEIGHT)

        screen_x = max(10, min(MONITOR_WIDTH - 10, screen_x))
        screen_y = max(10, min(MONITOR_HEIGHT - 10, screen_y))

        if mouse_control_enabled:
            with mouse_lock:
                mouse_target[:] = [screen_x, screen_y]

    # -------- Controls --------
    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        time.sleep(0.3)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_offset_yaw = 180 - raw_yaw
        calibration_offset_pitch = 180 - raw_pitch
        print("[Calibrated]")


cap.release()
cv2.destroyAllWindows()
