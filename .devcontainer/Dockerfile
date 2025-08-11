# Use official Python image as base
FROM python:3.10-slim

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install whisper
RUN pip install --upgrade pip
RUN pip install openai-whisper