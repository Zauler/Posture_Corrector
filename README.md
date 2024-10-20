# Posture Detection System

This project aims to detect bad posture using computer vision and machine learning techniques. By leveraging **MediaPipe** and **OpenCV**, the system analyzes key points on the human body, such as the nose and shoulders, to determine whether a person maintains a good or bad posture. The detection is based on calculating the angles between these key points and comparing them with predefined thresholds.

## Project Description

The **Posture Detection System** captures live video from a webcam and processes the video frames to extract human body landmarks using **MediaPipe's Pose** solution. With this landmark data, the program calculates specific angles between critical points (such as shoulders and neck) to identify potential posture problems. If the angles deviate beyond certain thresholds, the system labels the posture as incorrect.

This project could be especially helpful for improving ergonomic awareness, maintaining a healthy posture during prolonged sitting or standing sessions, and preventing back or neck pain due to poor posture habits.

## Technology Used

The main technologies and libraries used in this project are:

- **Python**: The main programming language.
- **OpenCV**: Used for video capture and image processing.
- **MediaPipe**: Provides body landmark detection to extract key points from the human pose.
- **NumPy**: Used for efficient numerical calculations, such as angle measurements.

## How It Works

1. **Video Capture**: The program captures a live feed from the webcam using **OpenCV**.
2. **Pose Detection**: The live frames are processed using **MediaPipe’s Pose** module to extract body landmarks such as the nose, left shoulder, and right shoulder.
3. **Angle Calculation**: The program calculates two main angles:
   - **Neck Flexion Angle**: Measures the angle between the nose and the midpoint of the shoulders with a vertical reference vector.
   - **Shoulder Alignment Angle**: Measures the angle between the line formed by the shoulders and a horizontal reference vector.
4. **Posture Evaluation**: Based on the calculated angles, the program determines if the posture is good or bad using predefined thresholds.
5. **Visual Feedback**: The program displays the detected posture status ("GOOD POSTURE" or "BAD POSTURE") on the video feed, along with the calculated angles.

## How It Detects Posture

1. **Neck Flexion Angle**: The system calculates the angle between the neck and the nose. If this angle exceeds 15 degrees, it indicates that the neck is flexed forward excessively.
2. **Shoulder Alignment Angle**: The system checks the horizontal alignment of the shoulders. If the shoulders are misaligned (angle deviates from 180 degrees by more than 10 degrees), it flags the posture as incorrect.
3. **Relative Distance Check**: The program compares the y-distance between the nose and neck with the distance between the shoulders to ensure that the head is not excessively forward.


