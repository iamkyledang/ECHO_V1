# AI Subtitle Generator

An automated subtitle generation system that combines Voice Activity Detection (VAD) with speech-to-text technology to generate SRT subtitle files from video content.

## Overview

This project provides two main functionalities:

1. **Training a Voice Activity Detection (VAD) Model** - Build and train a VAD model on speech data
2. **Running the Subtitle Generator** - Use the trained VAD model to automatically generate subtitle files from TED-talk videos

## Features

- Voice Activity Detection model training using the Voxconverse dataset
- Automatic subtitle generation in SRT format
- GUI-based subtitle generator software
- Evaluation tools for model performance and subtitle quality assessment

## Project Structure

```
├── prepare_voxconverse.py          # Data preprocessing for Voxconverse dataset
├── standalone_train_vad.py         # VAD model training pipeline
├── standalone_inference.py         # Example inference script
├── main.py                         # Main subtitle generator GUI
├── transcriber.py                  # Speech-to-text and VAD module
├── evaluated_vad_model.py          # VAD model evaluation script
├── srt_evaluation.py               # SRT subtitle quality evaluation
└── cleaned_dataset_MFA/            # Prepared dataset directory
```

## Usage

### Training the VAD Model

1. **Prepare the dataset:**
   ```bash
   python prepare_voxconverse.py
   ```
   This script formats the Voxconverse dataset for training.

2. **Train the model:**
   ```bash
   python standalone_train_vad.py
   ```
   This generates the `model.pt` file containing your trained VAD model.

3. **Test inference (optional):**
   ```bash
   python standalone_inference.py
   ```
   Example script demonstrating how to use the trained model.

### Running the Subtitle Generator

1. **Start the application:**
   ```bash
   python main.py
   ```
   This launches the GUI-based subtitle generator.

2. **How it works:**
   - The `main.py` file imports and uses the `transcriber.py` module
   - The trained VAD model is integrated into `transcriber.py` as a module
   - The system processes TED-talk video files and generates SRT subtitle files

## Evaluation

### Model Evaluation
Use `evaluated_vad_model.py` to evaluate the performance of the trained VAD model.

### Subtitle Quality Evaluation
Use `srt_evaluation.py` to assess the quality of generated SRT files by comparing them against manually created reference subtitles.

## Demo

A demonstration video is included in the project. Please review it to understand how to use the GUI software.

## Notes

- The system is optimized for TED-talk video content
- Manual subtitle files should be in SRT format for evaluation purposes
- Ensure the Voxconverse dataset is properly formatted before training