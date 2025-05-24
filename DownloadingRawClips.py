import pandas as pd
import os
import yt_dlp
import re

# Path to your CSV
csv_path = "youtube_scraped.csv"

# Output folder for downloaded videos
video_dir = "downloaded_videos"
os.makedirs(video_dir, exist_ok=True)

# Read CSV with proper column names
df = pd.read_csv(csv_path)
df = df[200:]
df = df[df['label'] != 'unknown']

# Helper to make filenames safe
def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s.strip().lower())[:50]

# Function to download a YouTube video
def download_video(link, out_path):
    ydl_opts = {
        'outtmpl': out_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([link])
            return True
        except Exception as e:
            print(f"❌ Failed to download {link} — {e}")
            return False

# Loop through each row and download
for idx, row in df.iterrows():
    title = row['title']
    url = row['url']
    label = row['label']

    # Build safe and unique filename
    safe_title = sanitize_filename(title)
    filename = f"{label}_{idx}_{safe_title}.mp4"
    output_path = os.path.join(video_dir, filename)

    print(f"⬇️ Downloading {title} ({label})")
    download_video(url, output_path)
