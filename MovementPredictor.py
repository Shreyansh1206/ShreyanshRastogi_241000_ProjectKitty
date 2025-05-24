import os
import sys
import cv2
import json
import torch
import subprocess
import numpy as np
import mediapipe as mp
import torch.nn as nn

# === Constants ===
MAX_FRAMES = 60
NUM_KEYPOINTS = 33
KEYPOINT_DIM = 2
INPUT_SIZE = NUM_KEYPOINTS * KEYPOINT_DIM
RESOLUTION = (640, 360)
FPS = 30

# === Load label map ===
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

idx_to_class = {v: k for k, v in label_map.items()}

# === Video standardization ===
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

# === Define LSTM Model ===
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

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPoseClassifier(INPUT_SIZE, 128, 64, len(label_map)).to(device)
model.load_state_dict(torch.load("lstm_classification_model.pth", map_location=device))
model.eval()

# === Initialize MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Keypoint extraction ===
def extract_keypoints_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [[lm.x, lm.y] for lm in landmarks]
        return np.array(keypoints)
    return None

# === Process standardized video ===
def process_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoint_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints_from_frame(frame)

        keypoint_sequence.append(keypoints[:, :2])

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

# === Run prediction ===
def predict(video_path):
    temp_path = "Input_video/standardized_deadlift_9.mp4"
    standardize_video(video_path, temp_path, RESOLUTION, FPS)

    sequence = process_clip(temp_path)
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        class_label = idx_to_class[predicted_class]

    os.remove(temp_path)
    return predicted_class

# === Entry Point ===
if __name__ == "__main__":
    video_file = "Input_video/Deadlift_7.mp4"  # Change to your input video
    predicted_class = predict(video_file)
