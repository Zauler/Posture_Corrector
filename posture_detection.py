import numpy as np
import cv2
import mediapipe as mp

# Function to calculate the angle between two 2D vectors
def vector_angle_2d(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors
    angle = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle)
    return angle_degrees

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        # Normalized coordinates of key points
        nose = [
            landmarks[mp_pose.PoseLandmark.NOSE.value].x,
            landmarks[mp_pose.PoseLandmark.NOSE.value].y
        ]

        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]

        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]

        # Neck point (midpoint between the shoulders)
        neck = [
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ]

        # Vectors for angle calculations
        neck_to_nose_vector = [
            nose[0] - neck[0],
            nose[1] - neck[1]
        ]

        shoulder_vector = [
            right_shoulder[0] - left_shoulder[0],
            right_shoulder[1] - left_shoulder[1]
        ]

        # Reference vectors
        vertical_vector = [0, -1]  # y-axis pointing up
        horizontal_vector = [1, 0]  # x-axis pointing right

        # Angle calculations
        neck_flexion_angle = vector_angle_2d(neck_to_nose_vector, vertical_vector)
        shoulder_angle = vector_angle_2d(shoulder_vector, horizontal_vector)

        # Convert to pixel coordinates for drawing
        height, width, _ = frame.shape
        nose_pixel = (int(nose[0] * width), int(nose[1] * height))
        left_shoulder_pixel = (int(left_shoulder[0] * width), int(left_shoulder[1] * height))
        right_shoulder_pixel = (int(right_shoulder[0] * width), int(right_shoulder[1] * height))
        neck_pixel = (int(neck[0] * width), int(neck[1] * height))

        # Draw points and lines
        cv2.circle(frame, nose_pixel, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_shoulder_pixel, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_shoulder_pixel, 5, (255, 0, 0), -1)
        cv2.circle(frame, neck_pixel, 5, (0, 255, 255), -1)
        cv2.line(frame, left_shoulder_pixel, right_shoulder_pixel, (255, 0, 0), 2)
        cv2.line(frame, neck_pixel, nose_pixel, (0, 255, 0), 2)

        # Display the angles on the screen
        cv2.putText(frame, f"Neck Angle: {neck_flexion_angle:.2f} degrees",
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.2f} degrees",
                    (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

        # Determine posture
        if neck_flexion_angle > 15 or shoulder_angle < 170 or ( abs(nose[1] - neck[1]) < (left_shoulder[0] - right_shoulder[0])/2 ):
            cv2.putText(frame, "BAD POSTURE",
                        (int(width * 0.05), int(height * 0.9)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "GOOD POSTURE",
                        (int(width * 0.05), int(height * 0.9)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()