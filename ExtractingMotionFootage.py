import numpy as np
import cv2
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import *
import os

print(type(os))

MOTION_THRESHOLD = 0.002
MIN_CONSECUTIVE_FRAMES = 15
KEYPOINT_VISIBILITY_THRESHOLD = 0.5
MAX_SEGMENTS = 50
video_folder = 'processed_downloaded_videos'
count = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

def extract_label_from_filename(filename):
    if "squat" in filename.lower():
        return "Squat"
    elif "deadlift" in filename.lower():
        return "Deadlift"
    elif "pull-up" in filename.lower() or "pull-up" in filename.lower():
        return "Pull-Up"
    return "Unknown"

def extract_keypoints(image):
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rotated_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
    results = pose.process(rotated_frame)
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility < 0.5:
                return None
            keypoints.append((landmark.x, landmark.y))
        return np.array(keypoints)
    return None

def get_motion_scores(video_path):
    cap = cv2.VideoCapture(video_path)
    motion_scores = []
    valid_frames = []

    prev_keypoints = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(frame)
        if keypoints is None:
            motion_scores.append(0)
            valid_frames.append(False)
            prev_keypoints = None
        else:
            valid_frames.append(True)
            if prev_keypoints is not None:
                motion = np.linalg.norm(keypoints - prev_keypoints)
                motion_scores.append(motion)
            else:
                motion_scores.append(0)
            prev_keypoints = keypoints

        frame_idx += 1
    cap.release()
    return np.array(motion_scores), np.array(valid_frames)

def find_motion_segments(motion_scores, valid_frames, fps):
    High_Motion = []
    for i in range(len(valid_frames)):
        if(valid_frames[i] and motion_scores[i] > MOTION_THRESHOLD):
            High_Motion.append(True)
        else:
            High_Motion.append(False)
    
    segments = []
    start = None
    low_motion_count = 0

    for i, is_high in enumerate(High_Motion):
        if is_high:
            if start is None:
                start = i
            low_motion_count = 0  # Reset the counter
        else:
            if start is not None:
                low_motion_count += 1
                if low_motion_count >= 15:  # Require 10 low-motion frames to end a segment
                    end = i - low_motion_count + 1  # End before the low-motion sequence
                    if end - start >= MIN_CONSECUTIVE_FRAMES:
                        segments.append((start, end))
                    start = None
                    low_motion_count = 0
    
    if start is not None and len(High_Motion) - start >= MIN_CONSECUTIVE_FRAMES:
        segments.append((start, len(High_Motion)))

    return [(start/fps, end/fps, start, end) for start, end in segments]
    
def extract_video_segment(input_path, output_path, start_time, end_time):
    clip = VideoFileClip(input_path)
    try:
        background = clip.subclipped(start_time, end_time)
        final_clip = CompositeVideoClip([background])
        final_clip.write_videofile(output_path, codec='libx264', audio=False, logger = None)
    except Exception as e:
        print(f"Error processing this one: {e}")
    

def extract_all_segments(video_file_path, output_dir, fps=30):
    global count

    os.makedirs(output_dir, exist_ok = True)

    motion_scores, valid_frames = get_motion_scores(f'{video_folder}\{video_file_path}')
    segments = find_motion_segments(motion_scores, valid_frames, fps)

    if not segments:
        print("No Valid stroke segments detected!!")
        return
    
    for idx, (start_time, end_time, start_frame, end_frame) in enumerate(segments[:MAX_SEGMENTS]):
        print(f"Saving segment {idx+1}: {start_time:.2f}s to {end_time:.2f}s")

        label = extract_label_from_filename(video_file_path)
        video_out = os.path.join(output_dir, f'{label}_{count}.mp4')
        count += 1
        extract_video_segment(f'{video_folder}\{video_file_path}', video_out, start_time, end_time)

        print(f"\n Extracted {min(len(segments), MAX_SEGMENTS)} segment(s)")


for filename in os.listdir(video_folder):
    extract_all_segments(filename, 'Output_segments', fps = 30)

