# ShreyanshRastogi_241000_ProjectKitty
## Project Kitty

The goal of Project KITTY is to develop a machine-learning model-based system capable of analysing human motion from video clips, to classify them into a type of motion, and to give corrective feedback or tactical insights related to the motion. 

## Features
- Keypoint extraction and sport-action classification
- Giving analysis and corrective measures for sport-action

## 📁 Project Structure
├── Dataset.creation/ # Scripts for data collection and preparation
│ ├── ConvertingRawFootageToSameFormat.py
│ ├── DataAugmentation.py
│ ├── DatasetFormation.py
│ ├── DownloadingRawClips.py
│ ├── ExtractingMotionFromClips.py
│ ├── ManualFiltering.py
│ └── RawClipsExtraction.py
├── Input_video/ # Raw input videos for prediction/analysis
├── Output_segments/ # Segmented output video clips
├── processed_downloaded/ # Processed videos/data ready for training
├── downloaded_videos/ # Downloaded raw videos
├── label_map.json # Maps action labels to class indices
├── ClassificationModel.py # LSTM model training file
├── lstm_classification_model.pth # Trained model weights
├── MovementPredictorAndAnalyzer.py # Predict movement on new video
├── testing.py # Inference script for testing
├── X_60.npy # Feature data (60-frame sequences)
├── y_60.npy # Corresponding labels
├── X_60_augmented.npy # Augmented training data
├── y_60_augmented.npy # Augmented labels
├── youtube_scraped.csv # Metadata of scraped videos

## Dependencies
All the dependencies are mentioned in the requirement.txt file attached in the repository.
Moreover, for the entire project, I've used a virtual environment in python version 3.10.16

## Usage
- For sure, load the label_map.json and lstm_classification_model.pth file for getting predictions and analysis.
- Create a folder named Input_video and enter your motion clip in the folder, let's say Video.mp4
- Open the MovementPredictorAndAnalyzer.py file, at the entry point of code towards the end of file, change the variable named video_file to your relative video file ("Input_video/Video.mp4" in this case)
- Run the code, you'll get the predicted class, as well as the analysis and corrective measures.
