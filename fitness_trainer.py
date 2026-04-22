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

def show_guide():
    guide = np.zeros((480, 640, 3), dtype=np.uint8)

    # Background
    guide[:] = (20, 20, 20)

    # Title box
    draw_box(guide, 0, 0, 640, 55, (0, 102, 102))
    cv2.putText(guide, 'AI FITNESS TRAINER - GUIDE', (80, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Bicep Curl box
    draw_box(guide, 20, 70, 290, 270, (30, 60, 30))
    cv2.putText(guide, 'BICEP CURL (Press B)', (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 1)
    cv2.line(guide, (30, 108), (300, 108), (0, 200, 100), 1)

    steps_bicep = [
        '1. Stand facing the camera',
        '2. Keep feet shoulder width apart',
        '3. Hold arms straight down',
        '4. Slowly curl arm UP to shoulder',
        '5. Hold for 1 second at top',
        '6. Slowly lower arm back DOWN',
        '7. Keep elbow close to your body',
        '8. Do not swing your elbow out',
        '',
        'COUNT: Down to Up to Down = 1 Rep',
    ]
    for i, step in enumerate(steps_bicep):
        cv2.putText(guide, step, (30, 128 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)

    # Squat box
    draw_box(guide, 330, 70, 290, 270, (60, 30, 30))
    cv2.putText(guide, 'SQUAT (Press S)', (340, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 255), 1)
    cv2.line(guide, (340, 108), (610, 108), (0, 100, 200), 1)

    steps_squat = [
        '1. Stand facing the camera',
        '2. Keep feet shoulder width apart',
        '3. Stand fully straight',
        '4. Slowly bend knees going DOWN',
        '5. Go down until knees are at 90',
        '6. Keep back straight always',
        '7. Do not let knees go past toes',
        '8. Slowly rise back UP again',
        '',
        'COUNT: Up to Down to Up = 1 Rep',
    ]
    for i, step in enumerate(steps_squat):
        cv2.putText(guide, step, (340, 128 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)

    # Warnings box
    draw_box(guide, 20, 355, 600, 70, (60, 20, 20))
    cv2.putText(guide, 'POSTURE WARNINGS YOU MAY SEE:', (30, 378),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    cv2.putText(guide, 'KEEP ELBOW STILL = Your elbow is swinging outward during curl. Fix your form.',
                (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)
    cv2.putText(guide, 'PLEASE FACE THE CAMERA = You are turned sideways. Face the camera directly.',
                (30, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)

    # Controls box
    draw_box(guide, 20, 435, 600, 35, (40, 40, 40))
    cv2.putText(guide, 'B = Bicep  |  S = Squat  |  R = Reset  |  Q = Quit',
                (60, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Press Enter
    draw_box(guide, 0, 455, 640, 25, (0, 102, 102))
    cv2.putText(guide, 'Press ENTER to Start Training',
                (170, 473), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    while True:
        cv2.imshow('AI Fitness Trainer', guide)
        key = cv2.waitKey(10) & 0xFF
        if key == 13:  # ENTER key
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()

counter = 0
stage = None
elbow_start_x = None
exercise = "bicep"

show_guide()

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        draw_box(image, 0, 0, 640, 50, (30, 30, 30))
        cv2.putText(image, 'AI FITNESS TRAINER', (180, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        draw_box(image, 0, 55, 220, 50, (50, 50, 50))
        cv2.putText(image, 'EXERCISE', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, exercise.upper(), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        draw_box(image, 225, 55, 180, 50, (50, 50, 50))
        cv2.putText(image, 'REPS', (235, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(counter), (270, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        draw_box(image, 410, 55, 220, 50, (50, 50, 50))
        cv2.putText(image, 'STAGE', (420, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, str(stage).upper() if stage else '-', (420, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
                if angle < 91 and stage == "up":
                    stage = "down"
                    counter += 2

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