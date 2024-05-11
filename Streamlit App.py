#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import json
import librosa
import pandas as pd
from pandas import DataFrame
import numpy as np
import torch
import torch.nn as nn
import requests
import base64

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set span
span = 1500

# GitHub repository information
username = "shivmv"
repository = "Heart-Murmurs"
base_url = f"https://github.com/{username}/{repository}/blob/main/"

# Function to download file from GitHub
def download_file_from_github(file_path):
    response = requests.get(f"{base_url}{file_path}")
    content = response.content
    content_decoded = base64.b64decode(content).decode('utf-8')
    return content_decoded

# Load thresholds
thresholds_content = download_file_from_github("thresholds.txt")
thresholds = json.loads(thresholds_content)

# Get list of model URLs
model_paths = [
    "Model_Abnormal.pth",
    "Model_Absent.pth",
    "Model_Normal.pth",
    "Model_Present.pth"
]

# Convert function
def convert(patient_input, audio_foldername=None):
    # Load and process the audio file
    wav, sr = librosa.load(patient_input, sr=44100, mono=True)
    wav = librosa.util.normalize(wav)
    solo_beat = librosa.resample(wav, orig_sr=sr, target_sr=span, fix=False)
    solo_beat = librosa.util.fix_length(solo_beat, size=span)
    solo_beat = librosa.util.normalize(solo_beat)

    return DataFrame(solo_beat).transpose()

# Create dataset function
def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features

# Predict function
def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses

custom_messages = {
    "Pathological_Present": "The murmur is pathological and present.",
    "Not Pathological_Present": "The murmur is not pathological but present.",
    "Pathological_Not Present": "The murmur is pathological but not present.",
    "Not Pathological_Not Present": "The murmur is not pathological and not present.",
}

def run_murmur_analysis(input_file):
    patient_audio_data = input_file.read()
    patient_audio_converted = convert(patient_audio_data, folder)
    patient_dataset, *_ = create_dataset(patient_audio_converted)

    overall_certainty = {}
    for model_path in model_paths:
        model_content = download_file_from_github(model_path)
        model = torch.load(model_content, map_location=device)
        model = model.to(device)
        reconstructed, mismatch = predict(model, patient_dataset)
        condition_name = model_path.split('/')[-1].split('.')[0]  # Extract condition name from URL
        certainty_value = 2**(-mismatch[0] / thresholds[condition_name])
        overall_certainty[condition_name] = certainty_value

    pathological_certainty = sum(overall_certainty.get("Pathological", 0))
    not_pathological_certainty = sum(overall_certainty.get("Not Pathological", 0))
    present_certainty = sum(overall_certainty.get("Present", 0))
    not_present_certainty = sum(overall_certainty.get("Not Present", 0))

    is_pathological = pathological_certainty > not_pathological_certainty
    is_present = present_certainty > not_present_certainty

    result = custom_messages.get(
        (
            f"{'Pathological' if is_pathological else 'Not Pathological'}_"
            f"{'Present' if is_present else 'Not Present'}"
        ),
        "No result found.")

    st.write(result)

st.title("Heart Murmur Analysis")

# File uploader
input_file = st.file_uploader("Upload a WAV file", type=["wav"])

if input_file is not None:
    run_murmur_analysis(input_file)
else:
    st.write("Please upload a WAV file to analyze.")

