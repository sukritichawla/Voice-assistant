import os
import numpy as np
import librosa
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===== CONFIG =====
MODEL_PATH = "model_2.h5"
LABELS_PATH = "labels_2.json"
DATASET_PATH = r"C:\Users\ADMIN\Desktop\Sukriti\VIT\SL\AnuAssistant\dataset"

SAMPLE_RATE = 22050
N_MFCC = 20
MAX_PAD_LEN = 40

# ===== LOAD MODEL =====
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

index_to_label = {v: k for k, v in labels.items()}

# ===== PREPROCESS =====
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    mfcc = mfcc.reshape(1, N_MFCC, MAX_PAD_LEN, 1)

    return mfcc

# ===== LOAD DATA =====
y_true = []
y_pred = []

print("📂 Evaluating dataset...")

for label_name, label_index in labels.items():
    folder_path = os.path.join(DATASET_PATH, label_name)

    for file in os.listdir(folder_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            try:
                features = preprocess_audio(file_path)

                prediction = model.predict(features, verbose=0)
                pred_index = np.argmax(prediction)

                y_true.append(label_index)
                y_pred.append(pred_index)

            except Exception as e:
                print(f"❌ Error: {file_path}")

# ===== RESULTS =====
print("\n🎯 Accuracy:")
print(accuracy_score(y_true, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(labels.keys())))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))