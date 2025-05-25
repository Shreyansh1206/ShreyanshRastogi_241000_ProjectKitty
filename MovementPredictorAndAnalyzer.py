import os
import sys
import cv2
import json
import torch
import subprocess
import numpy as np
import mediapipe as mp
import torch.nn as nn

# Constants
MAX_FRAMES = 60
NUM_KEYPOINTS = 33
KEYPOINT_DIM = 2
INPUT_SIZE = NUM_KEYPOINTS * KEYPOINT_DIM
RESOLUTION = (640, 360)
FPS = 30

#Label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

idx_to_class = {v: k for k, v in label_map.items()}

# Video Standardization
def standardize_video(input_path, output_path, resolution=(640, 360), fps=30):
    width, height = resolution
    cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vf', f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-strict', '-2',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Defining LSTM Model
class LSTMPoseClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super(LSTMPoseClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(MAX_FRAMES)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden2, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, len(label_map))

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out

#Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPoseClassifier(INPUT_SIZE, 128, 64, len(label_map)).to(device)
model.load_state_dict(torch.load("lstm_classification_model.pth", map_location=device))
model.eval()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Keypoints extraction frame wise
def extract_keypoints_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [[lm.x, lm.y, lm.z] for lm in landmarks]
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
            keypoint_sequence.append(keypoints[:, :2])  # Only (x, y)

    cap.release()

    if len(keypoint_sequence) == 0:
        print("❌ Error: No valid frames with keypoints found.")
        sys.exit(1)

    sequence = np.array(keypoint_sequence)
    T = sequence.shape[0]

    if T > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    elif T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, NUM_KEYPOINTS, KEYPOINT_DIM))
        sequence = np.concatenate((sequence, pad), axis=0)

    return sequence.reshape(MAX_FRAMES, INPUT_SIZE)

def process_clip_3d(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoint_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [[lm.x, lm.y, lm.z] for lm in landmarks]
            keypoint_sequence.append(np.array(keypoints))  # (33, 3)

    cap.release()

    if len(keypoint_sequence) == 0:
        print("❌ Error: No valid frames with keypoints found.")
        sys.exit(1)

    sequence = np.array(keypoint_sequence)  # shape: (T, 33, 3)
    T = sequence.shape[0]

    if T > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    elif T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, NUM_KEYPOINTS, 3))
        sequence = np.concatenate((sequence, pad), axis=0)

    return sequence  # shape: (60, 33, 3)

def predict_and_analyse(video_path):
    temp_path = "Input_video/temp.mp4"
    standardize_video(video_path, temp_path, RESOLUTION, FPS)

    sequence = process_clip(temp_path)
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        class_label = idx_to_class[predicted_class]

    label = idx_to_class[predicted_class]
    print(f"Predicted Motion: {label}")

    sequence3d = process_clip_3d(temp_path)
    if(predicted_class == 0):
        feedback_deadlift(sequence3d)
    if(predicted_class == 1):
        feedback_pullup(sequence3d)
    if(predicted_class == 2):
        feedback_squat(sequence3d)
    os.remove(temp_path)
    

def calculate_angle_3d(a, b, c):
    # a, b, c: np.array of shape (3,)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def feedback_deadlift(landmark_sequence):
    knee_angles, back_angles = [], []

    for frame in landmark_sequence:
        hip = frame[23]
        knee = frame[25]
        ankle = frame[27]
        shoulder = frame[11]

        knee_angle = calculate_angle_3d(hip, knee, ankle)
        back_angle = calculate_angle_3d(shoulder, hip, knee)

        knee_angles.append(knee_angle)
        back_angles.append(back_angle)

    min_knee = min(knee_angles)
    max_back_deviation = max([abs(180 - angle) for angle in back_angles])
    print(f"Minimum Knee Angle: {min_knee:.1f}°")
    print(f"Maximum Back Deviation: {max_back_deviation:.1f}°")

    feedback = []
    if min_knee < 80:
        feedback.append("❌ Bend your knees less during the deadlift.")
    else:
        feedback.append("✅ Knee bending is within a good range.")

    if max_back_deviation > 150:
        feedback.append("❌ Keep your back straighter throughout the lift.")
    else:
        feedback.append("✅ Back straightness looks good.")

    for fb in feedback:
        print(fb)

def feedback_pullup(landmark_sequence):
    elbow_angles, torso_angles = [], []

    for frame in landmark_sequence:
        shoulder = frame[11]
        elbow = frame[13]
        wrist = frame[15]
        hip = frame[23]

        elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)
        torso_angle = calculate_angle_3d(shoulder, hip, wrist)

        elbow_angles.append(elbow_angle)
        torso_angles.append(torso_angle)

    max_elbow = max(elbow_angles)
    max_torso_deviation = max([abs(180 - angle) for angle in torso_angles])
    print(f"Max Elbow Angle: {max_elbow:.1f}°")
    print(f"Maximum Torso deviation: {max_torso_deviation:.1f}°")

    feedback = []
    if max_elbow > 170:
        feedback.append("❌ Avoid locking your elbows at the bottom of the pull-up.")
    else:
        feedback.append("✅ Elbow range is controlled.")

    if max_torso_deviation > 170:
        feedback.append("❌ Reduce body swing/arching during pull-ups.")
    else:
        feedback.append("✅ Core control looks good.")

    for fb in feedback:
        print(fb)

def feedback_squat(landmark_sequence):
    knee_angles, torso_angles = [], []

    for frame in landmark_sequence:
        hip = frame[23]
        knee = frame[25]
        ankle = frame[27]
        shoulder = frame[11]

        knee_angle = calculate_angle_3d(hip, knee, ankle)
        torso_angle = calculate_angle_3d(shoulder, hip, ankle)

        knee_angles.append(knee_angle)
        torso_angles.append(torso_angle)

    min_knee = min(knee_angles)
    max_torso_deviation = max([abs(180 - angle) for angle in torso_angles])
    print(f"Minimum Knee Angle: {min_knee:.1f}°")
    print(f"Maximum Torso Deviation: {max_torso_deviation:.1f}°")

    feedback = []
    if min_knee < 80:
        feedback.append("❌ You're squatting too deep. Try to stay above 80°.")
    else:
        feedback.append("✅ Squat depth is appropriate.")

    if max_torso_deviation > 25:
        feedback.append("❌ Keep your chest more upright during squats.")
    else:
        feedback.append("✅ Chest posture is upright.")

    for fb in feedback:
        print(fb)

# === Entry Point ===
if __name__ == "__main__":
    video_file = "Input_video/video.mp4"  # Change to your input video
    predict_and_analyse(video_file)

