import os
import numpy as np
import librosa
import soundfile as sf

DATASET_PATH = r"C:\Users\ADMIN\Desktop\Sukriti\VIT\SL\AnuAssistant\dataset"

def add_noise(audio):
    noise = np.random.randn(len(audio))
    return audio + 0.005 * noise

def time_stretch(audio):
    rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr):
    steps = np.random.randint(-3, 3)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def time_shift(audio):
    shift = int(np.random.uniform(-0.2, 0.2) * len(audio))
    return np.roll(audio, shift)

def change_volume(audio):
    factor = np.random.uniform(0.5, 1.5)
    return audio * factor

def normalize(audio):
    if np.max(np.abs(audio)) == 0:
        return audio
    return audio / np.max(np.abs(audio))

total_files = 0
augmented_files = 0

for root, dirs, files in os.walk(DATASET_PATH):
    print(f"\n📂 Checking folder: {root}")

    for file in files:
        if file.lower().endswith(".wav"):   # FIXED
            total_files += 1

            file_path = os.path.join(root, file)
            print(f"🎧 Processing: {file_path}")

            try:
                audio, sr = librosa.load(file_path, sr=None)

                augmented_versions = {
                    "noise": add_noise(audio),
                    "stretch": time_stretch(audio),
                    "pitch": pitch_shift(audio, sr),
                    "shift": time_shift(audio),
                    "volume": change_volume(audio)
                }

                for aug_name, aug_audio in augmented_versions.items():
                    aug_audio = normalize(aug_audio)

                    new_filename = file.replace(".wav", f"_{aug_name}.wav")
                    new_path = os.path.join(root, new_filename)

                    sf.write(new_path, aug_audio, sr)
                    augmented_files += 1

                    print(f"   ✅ Saved: {new_filename}")

            except Exception as e:
                print(f"❌ Error: {e}")

print("\n========================")
print(f"Total original files: {total_files}")
print(f"Total augmented files created: {augmented_files}")
print("🎉 DONE")