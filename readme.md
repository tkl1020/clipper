# Clipper v2.0 - AI Video Highlight Detection

Clipper is an AI-powered video editor that automatically detects interesting moments in your videos and helps you create highlight clips with just a few clicks.

## Features

- **Video playback** with standard controls (play, pause, seek)
- **AI-powered transcription** of video/audio content
- **Automatic detection of highlights** based on emotional content in speech
- **Manual and automatic clip editing** tools
- **Live typing animation** for transcripts
- **Elegant dark theme** for comfortable video editing
- **One-click export** of clips and transcripts

## Installation

### Prerequisites

- Python 3.7 or higher
- FFMPEG (required for video processing)
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via Chocolatey
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg` or `sudo dnf install ffmpeg`

### Setup Instructions

1. Clone this repository or download the script:
   ```
   git clone https://github.com/yourusername/clipper.git
   cd clipper
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv clipper-env
   source clipper-env/bin/activate  # On Windows: clipper-env\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python clipper-v2.0.py
   ```

## Usage Guide

### Loading a Video

1. Click the "Load Video" button
2. Select a video file (supported formats: MP4, MOV, AVI, MKV)
3. The video will load and be ready for playback

### Basic Playback

- Use the play/pause button to control playback
- Drag the slider to navigate through the video
- Adjust volume using the volume slider

### Creating Clips Manually

1. Navigate to the desired start point in the video
2. Click "Mark Start"
3. Navigate to the desired end point
4. Click "Mark End"
5. Click "Save Clip" to export the clip

### Using AI Highlight Detection

1. Load a video
2. Click "Transcribe + Detect Highlights"
3. Wait for the AI to process the video (this may take some time depending on the length)
4. Review the detected highlights in the right panel
5. Double-click on a highlight to jump to that point in the video
6. Use the "CUT CLIP" button to directly save the highlighted section

### Manual Time Entry

For precise control, you can manually enter timecodes:
1. Type the start time in HH:MM:SS format
2. Type the end time in HH:MM:SS format
3. Click "Apply Manual Times"

### Saving Transcripts

After transcription is complete, click "Transcript" to save the full transcript as a text file.

## How It Works

Clipper uses two AI models:
1. **OpenAI's Whisper** for accurate speech-to-text transcription
2. **DistilBERT emotion classifier** to detect emotional content in speech

The application identifies moments of joy, surprise, and other emotions that often represent interesting highlights in videos.

## System Requirements

- **Minimum**: 4GB RAM, 2GB free disk space
- **Recommended**: 8GB+ RAM, modern multi-core CPU
- **Optional**: CUDA-compatible GPU for faster AI processing

## Known Limitations

- Processing large videos (1hr+) may require significant memory and processing time
- Currently optimized for English language content
- First-time use requires internet connection to download AI models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- Hugging Face for the Transformers library
- The PyQt team for the GUI framework
- MoviePy developers for the video editing functionality