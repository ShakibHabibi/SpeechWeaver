# SpeechWeaver

A powerful speech-to-text application that automatically types your transcribed speech into any text input field across your operating system. Using OpenAI's Whisper model with a system tray interface, SpeechWeaver seamlessly integrates with your workflow by automatically entering transcribed text wherever your cursor is focused.

> **Note**: SpeechWeaver is currently only available for Linux systems, with Ubuntu being the primary supported platform. Windows and macOS support are not available at this time.

## Features

- Automatically types transcribed text into any focused text field across your operating system
- Real-time speech recognition using OpenAI's Whisper model
- System tray interface for easy control
- Support for multiple Whisper models (tiny to large-v3)
- Clipboard support for copying transcriptions
- Memory optimization for long recordings
- CUDA support for GPU acceleration (if available)
- Parallel processing of audio chunks for improved performance

## Key Differentiator

What sets SpeechWeaver apart from other speech-to-text applications is its seamless integration with your workflow. Instead of just transcribing speech to a separate window or clipboard, SpeechWeaver automatically types the transcribed text directly into whatever text field you're currently focused on. Whether you're writing an email, filling out a form, or coding in your IDE, SpeechWeaver will automatically enter the transcribed text right where you need it.

## Demo Video

Watch SpeechWeaver in action! [Watch Demo](demo.mp4)

## Alternative to Wispr Flow

SpeechWeaver serves as an alternative to Wispr Flow for Linux users. While Wispr Flow is only available for Mac and Windows, SpeechWeaver brings similar functionality to the Linux ecosystem with some key advantages:

- Open-source and free to use
- Direct integration with Linux systems
- No subscription required
- Built on OpenAI's Whisper model for high accuracy
- Customizable through the system tray interface

## Prerequisites

- Python 3.8 or higher
- Linux operating system (Ubuntu recommended, other distributions may work but are not officially supported)
- NVIDIA GPU (optional, for CUDA support)
- Ubuntu-specific dependencies (automatically installed by setup script)

## Quick Setup

1. Clone this repository:
```bash
git clone https://github.com/ShakibHabibi/SpeechWeaver.git
cd SpeechWeaver
```

2. Make the setup script executable and run it:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install all required Ubuntu system dependencies
- Create a Python virtual environment
- Install all Python dependencies
- Set up CUDA support if available
- Configure the application

## Manual Setup (if needed)

If you prefer to set up manually or if the setup script doesn't work for your system:

1. Install Ubuntu system dependencies:
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

- `Alt+,`: Start recording
- `Alt+.`: Stop recording and transcribe
- `Alt+/`: Cancel current process

### System Tray Menu

The application provides a system tray icon that changes color based on the current state:
- ⚫ Gray: Application is idle and ready to record
- 🟢 Green: Recording in progress
- 🔵 Blue: Processing audio (transcribing)

The system tray icon provides the following options:
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