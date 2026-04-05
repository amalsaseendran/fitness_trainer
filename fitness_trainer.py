import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def draw_box(image, x, y, w, h, color):
    cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)

counter = 0
stage = None
elbow_start_x = None
exercise = "bicep"

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- TOP HEADER BOX ---
        draw_box(image, 0, 0, 640, 50, (30, 30, 30))
        cv2.putText(image, 'AI FITNESS TRAINER', (180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- EXERCISE BOX ---
        draw_box(image, 0, 55, 220, 50, (50, 50, 50))
        cv2.putText(image, 'EXERCISE', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, exercise.upper(), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        # --- REPS BOX ---
        draw_box(image, 225, 55, 180, 50, (50, 50, 50))
        cv2.putText(image, 'REPS', (235, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(counter), (270, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- STAGE BOX ---
        draw_box(image, 410, 55, 220, 50, (50, 50, 50))
        cv2.putText(image, 'STAGE', (420, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(stage).upper() if stage else '-', (420, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- CONTROLS BOX AT BOTTOM ---
        draw_box(image, 0, 430, 640, 50, (30, 30, 30))
        cv2.putText(image, 'B = Bicep  |  S = Squat  |  R = Reset  |  Q = Quit',
                    (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_diff = abs(left_shoulder.z - right_shoulder.z)

            if shoulder_diff > 0.2:
                draw_box(image, 0, 380, 640, 40, (0, 0, 180))
                cv2.putText(image, 'PLEASE FACE THE CAMERA!', (130, 408),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if exercise == "bicep":
                shoulder = [left_shoulder.x, left_shoulder.y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                if elbow_start_x is None:
                    elbow_start_x = elbow[0]

                elbow_drift = abs(elbow[0] - elbow_start_x)
                if elbow_drift > 0.08:
                    draw_box(image, 0, 340, 640, 40, (0, 0, 180))
                    cv2.putText(image, 'KEEP ELBOW STILL!', (180, 368),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                if angle < 40 and stage == "down":
                    stage = "up"
                    counter += 1

            elif exercise == "squat":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('AI Fitness Trainer', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            exercise = "bicep"
            counter = 0
            stage = None
            elbow_start_x = None
        elif key == ord('s'):
            exercise = "squat"
            counter = 0
            stage = None
            elbow_start_x = None
        elif key == ord('r'):
            counter = 0
            stage = None
            elbow_start_x = None

cap.release()
cv2.destroyAllWindows()