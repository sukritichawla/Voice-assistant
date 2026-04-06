import numpy as np
import librosa
import json
from tensorflow.keras.models import load_model

# ===== CONFIG =====
MODEL_PATH = "model_2.h5"
LABELS_PATH = "labels_2.json"

SAMPLE_RATE = 22050
N_MFCC = 20
MAX_PAD_LEN = 40

# ===== LOAD MODEL =====
model = load_model(MODEL_PATH)

# ===== LOAD LABELS =====
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# Reverse mapping: index → label
index_to_label = {v: k for k, v in labels.items()}

print("✅ Model and labels loaded")

# ===== FUNCTION: PROCESS AUDIO =====
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    # Normalize (same as training)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # Reshape for CNN
    mfcc = mfcc.reshape(1, N_MFCC, MAX_PAD_LEN, 1)

    return mfcc

# ===== TEST FUNCTION =====
def predict_audio(file_path):
    print(f"\n🎧 Testing file: {file_path}")

    features = preprocess_audio(file_path)

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_label = index_to_label[predicted_index]

    print(f"🧠 Predicted: {predicted_label}")
    print(f"📊 Confidence: {confidence*100:.2f}%")

# ===== INPUT =====
file_path = input("\nEnter path of .wav file: ").strip()

predict_audio(file_path)