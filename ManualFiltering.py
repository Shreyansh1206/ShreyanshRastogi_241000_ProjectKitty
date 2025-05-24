import os
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import *

# ======== CONFIGURATION ========
VIDEO_FOLDER = "./Output_segments"  # Path to your extracted clips

def clip_video(original_path, start_time, end_time):
    try:
        with VideoFileClip(original_path) as video:
            background = video.subclipped(start_time, end_time)
            final_clip = CompositeVideoClip([background])
            final_clip.write_videofile(original_path, codec='libx264', audio=False, logger = None)
        print(f"✂️ Clipped and saved: {os.path.basename(original_path)}")
    except Exception as e:
        print(f"❌ Error clipping video: {e}")

def play_video(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"⚠️ Failed to open: {filepath}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute timestamp
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        time_secs = frame_no / fps
        mins, secs = divmod(int(time_secs), 60)
        millis = int((time_secs % 1) * 1000)
        timestamp = f"{int(mins):02}:{int(secs):02}.{millis:03}"

        # Add timestamp text to frame
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display frame
        cv2.imshow("Review Clip", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def review_videos(folder):
    files = sorted([f for f in os.listdir(folder)])
    i = 0

    while i < len(files):
        fname = files[i]
        fpath = os.path.join(folder, fname)

        print(f"\n▶️ Reviewing: {fname}")
        play_video(fpath)

        while True:
            decision = input("Keep this clip? (y/n/r/c): ").strip().lower()

            if decision == 'y':
                print(f"✅ Kept: {fname}")
                i += 1
                break
            elif decision == 'n':
                os.remove(fpath)
                print(f"🗑️ Deleted: {fname}")
                i += 1
                break
            elif decision == 'r':
                print(f"🔁 Replaying: {fname}")
                play_video(fpath)  # Replay current file
            elif decision == 'c':
                print(f"✂️ Clipping: {fname}")
                try:
                    start = input("Enter start time (in seconds or HH:MM:SS): ").strip()
                    end = input("Enter end time (in seconds or HH:MM:SS): ").strip()
                    clip_video(fpath, start, end)
                    i += 1
                    break
                except Exception as e:
                    print(f"⚠️ Failed to clip: {e}")
            else:
                print("❓ Invalid input. Use 'y' (keep), 'n' (delete), 'r' (replay current), or 'c' (clip footage)")

if __name__ == "__main__":
    review_videos(VIDEO_FOLDER)


