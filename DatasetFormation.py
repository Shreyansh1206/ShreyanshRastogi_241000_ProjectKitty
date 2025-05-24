import cv2
import numpy as np
import mediapipe as mp
import os
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

MAX_FRAMES = 60  # Fixed length for LSTM input
NUM_KEYPOINTS = 33  # MediaPipe Pose
KEYPOINT_DIM = 2  # x, y

def extract_keypoints_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        for lm in landmarks:
            keypoints.append([lm.x, lm.y])
        return np.array(keypoints)
    return None

def process_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoint_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints_from_frame(frame)
        if keypoints is not None:
            keypoint_sequence.append(keypoints[:, :2])  # Use only (x, y)

    cap.release()
    
    # Normalize sequence length
    if len(keypoint_sequence) == 0:
        return None  # Skip if no valid data

    sequence = np.array(keypoint_sequence)  # (T, 33, 2)
    T = sequence.shape[0]

    if T > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    elif T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, NUM_KEYPOINTS, KEYPOINT_DIM))
        sequence = np.concatenate((sequence, pad), axis=0)
    SEQUENCE = sequence.reshape(MAX_FRAMES, NUM_KEYPOINTS * KEYPOINT_DIM)
    return SEQUENCE

def extract_label_from_filename(filename):
    if "squat" in filename.lower():
        return "Squat"
    elif "deadlift" in filename.lower():
        return "Deadlift"
    elif "pull-up" in filename.lower() or "pull-up" in filename.lower():
        return "Pull-Up"
    return "Unknown"

def create_dataset_from_clips(video_folder):
    X = []
    y = []
    label_map = {}
    label_counter = 0

    for filename in os.listdir(video_folder):
        label = extract_label_from_filename(filename)
        print(label)
        if label == "Unknown":
            continue

        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1

        print(f"⏳ Processing {filename} → {label}")
        clip_path = os.path.join(video_folder, filename)
        sequence = process_clip(clip_path)

        if sequence is not None:
            X.append(sequence)
            y.append(label_map[label])
    
    X = np.array(X)  # Shape: (num_samples, MAX_FRAMES, NUM_KEYPOINTS*2)
    y = np.array(y)

    # Save dataset
    np.save("X_60.npy", X)
    np.save("y_60.npy", y)
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    print(f"\n✅ Saved {len(X)} sequences with shape {X.shape}")

# Usage
create_dataset_from_clips("Output_segments")