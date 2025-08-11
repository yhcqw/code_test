#!/bin/bash
# Install ffmpeg
sudo apt-get update
sudo apt-get install -y ffmpeg

# Install Whisper and dependencies
pip install --upgrade pip
pip install openai-whisper
