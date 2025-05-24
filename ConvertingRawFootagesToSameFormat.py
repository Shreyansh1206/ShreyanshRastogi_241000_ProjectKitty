import os
import subprocess

def standardize_video(input_path, output_path, resolution=(640, 360), fps=30):
    width, height = resolution
    cmd = [
        'ffmpeg',
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

def batch_convert_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            standardize_video(input_path, output_path)
            print(f"✅ Converted: {file}")

batch_convert_videos('downloaded_videos', 'processed_downloaded_videos')
