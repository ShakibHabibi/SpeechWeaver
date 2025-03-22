# SpeechWeaver

A real-time speech-to-text application using OpenAI's Whisper model with a system tray interface.

## Features

- Real-time speech recognition using OpenAI's Whisper model
- System tray interface for easy control
- Support for multiple Whisper models (tiny to large-v3)
- Automatic text typing
- Clipboard support for copying transcriptions
- Memory optimization for long recordings
- CUDA support for GPU acceleration (if available)
- Parallel processing of audio chunks for improved performance

## Prerequisites

- Python 3.8 or higher
- Linux operating system (tested on Ubuntu/Debian)
- NVIDIA GPU (optional, for CUDA support)

## Quick Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/SpeechWeaver.git
cd SpeechWeaver
```

2. Make the setup script executable and run it:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install all required system dependencies
- Create a Python virtual environment
- Install all Python dependencies
- Set up CUDA support if available
- Configure the application

## Manual Setup (if needed)

If you prefer to set up manually or if the setup script doesn't work for your system:

1. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip portaudio19-dev ffmpeg xclip
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate
```

2. Run the application:
```bash
python speechweaver.py
```

### Keyboard Shortcuts

- `Ctrl+1`: Start recording
- `Ctrl+2`: Stop recording and transcribe
- `Ctrl+3`: Cancel current process

### System Tray Menu

The application provides a system tray icon with the following options:
- Model selection (tiny to large-v3)
- Start/Stop recording
- Copy last transcription
- Copy last error
- Restart application
- Exit

## Performance Optimization

SpeechWeaver implements several performance optimizations to ensure smooth real-time transcription:

- **Parallel Processing**: The application splits audio input into manageable chunks and processes them concurrently, significantly reducing transcription latency.
- **Memory Management**: Efficient memory handling for long recordings by processing audio in chunks and clearing memory after each transcription.
- **GPU Acceleration**: Automatic utilization of CUDA when available for faster processing.

## Notes

- The application uses the "base" Whisper model by default. You can change the model through the system tray menu.
- For best performance with GPU support, ensure you have the latest NVIDIA drivers installed.
- The application automatically optimizes memory usage during long recordings.
- Clipboard functionality requires either `xclip` or `xsel` to be installed.
- The parallel processing feature automatically adjusts chunk size based on your system's capabilities.

## Troubleshooting

If you encounter any issues:

1. Check the error messages in the terminal
2. Ensure all system dependencies are installed
3. Verify that your microphone is properly connected and selected as the default input device
4. For GPU support, ensure you have the correct NVIDIA drivers installed

## License

[Your chosen license] 