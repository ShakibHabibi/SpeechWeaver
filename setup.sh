#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up SpeechWeaver...${NC}"

# Update package lists if using apt
if command_exists apt-get; then
    echo -e "${YELLOW}Updating package lists...${NC}"
    sudo apt-get update
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system package
install_package() {
    if command_exists apt-get; then
        sudo apt-get install -y "$1"
    elif command_exists yum; then
        sudo yum install -y "$1"
    elif command_exists dnf; then
        sudo dnf install -y "$1"
    else
        echo -e "${RED}Could not detect package manager. Please install $1 manually.${NC}"
        exit 1
    fi
}

# Check Python version
if ! command_exists python3; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Check pip
if ! command_exists pip3; then
    echo -e "${YELLOW}Installing pip...${NC}"
    if command_exists apt-get; then
        sudo apt-get install -y python3-pip
    elif command_exists yum; then
        sudo yum install -y python3-pip
    elif command_exists dnf; then
        sudo dnf install -y python3-pip
    else
        echo -e "${RED}Could not install pip. Please install pip manually.${NC}"
        exit 1
    fi
fi

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"

# Install portaudio (required for sounddevice)
if ! command_exists portaudio; then
    install_package "portaudio19-dev"
fi

# Install ffmpeg (required for whisper)
if ! command_exists ffmpeg; then
    install_package "ffmpeg"
fi

# Install xclip or xsel for clipboard support
if ! command_exists xclip && ! command_exists xsel; then
    echo -e "${YELLOW}Installing clipboard support...${NC}"
    install_package "xclip"
fi

# Create and activate virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Check CUDA availability
if command_exists nvidia-smi; then
    echo -e "${GREEN}NVIDIA GPU detected. CUDA support will be enabled.${NC}"
else
    echo -e "${YELLOW}No NVIDIA GPU detected. The application will run on CPU only.${NC}"
fi

# Make the main script executable
chmod +x speechweaver.py

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}To run SpeechWeaver:${NC}"
echo -e "1. Activate the virtual environment: ${GREEN}source venv/bin/activate${NC}"
echo -e "2. Run the application: ${GREEN}python speechweaver.py${NC}"
echo -e "\n${YELLOW}Keyboard shortcuts:${NC}"
echo -e "Alt+,: Start recording"
echo -e "Alt+.: Stop recording and transcribe"
echo -e "Alt+/: Cancel current process" 