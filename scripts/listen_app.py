import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import pickle
import soundfile as sf
import os
import time

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ======================
# LOAD MODEL
# ======================
print("Loading Anu Assistant...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

print("✅ Assistant started (Listening for wake word: ANU)")

# ======================
# AUDIO SETTINGS
# ======================
MIC_SR = 44100
MODEL_SR = 16000
WAKE_DURATION = 1.2
CMD_DURATION = 2.5

sd.default.device = 1   # your mic index


# ======================
# PREPROCESS
# ======================
def preprocess_audio(audio, sr):

    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    TARGET_SAMPLES = 16000
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    else:
        audio = audio[:TARGET_SAMPLES]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    TARGET_LENGTH = 64
    if mfcc.shape[1] < TARGET_LENGTH:
        mfcc = np.pad(mfcc, ((0,0),(0,TARGET_LENGTH-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :TARGET_LENGTH]

    return mfcc[np.newaxis, ..., np.newaxis]


# ======================
# RECORD AUDIO
# ======================
def record(seconds):

    audio = sd.rec(
        int(seconds * MIC_SR),
        samplerate=MIC_SR,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    return audio.flatten()


# ======================
# PREDICT
# ======================
def predict(audio):

    sf.write("temp.wav", audio, MIC_SR)
    audio, sr = librosa.load("temp.wav", sr=MODEL_SR)

    X = preprocess_audio(audio, sr)

    pred = model.predict(X, verbose=0)

    idx = np.argmax(pred)
    label = encoder.inverse_transform([idx])[0]
    conf = pred[0][idx]

    return label, conf


# ======================
# MAIN LOOP
# ======================
try:
    while True:

        # ---- WAKE WORD MODE ----
        audio = record(WAKE_DURATION)
        label, conf = predict(audio)

        if label == "anu" and conf > 0.7:
            print("\n🟢 Wake word detected!")

            # ---- COMMAND MODE ----
            print("🎤 Listening for command...")
            cmd_audio = record(CMD_DURATION)

            cmd_label, cmd_conf = predict(cmd_audio)

            print(f"🧠 Command Detected: {cmd_label} ({cmd_conf*100:.1f}%)")
            print("🔵 Returning to wake listening...\n")

except KeyboardInterrupt:
    print("\n🛑 Assistant stopped.")
