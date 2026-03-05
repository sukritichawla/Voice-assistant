import os
import librosa
import numpy as np

DATASET_PATH = "dataset"

checked = 0

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            path = os.path.join(folder, file)

            try:
                audio, sr = librosa.load(path, sr=16000)

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=40
                )

                print(f"✅ OK: {label}/{file} | shape={mfcc.shape}")

                checked += 1
                break   # check only 1 file per label

            except Exception as e:
                print(f"❌ ERROR: {path}", e)

print(f"\nChecked {checked} labels successfully.")
