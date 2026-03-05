import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import time

# ---------------- DEVICE ----------------
sd.default.device = 1

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ---------------- LOAD MODEL ----------------
print("Loading assistant...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

print("✅ Assistant running (Say 'anu' to wake)")

# ---------------- AUDIO SETTINGS ----------------
MODEL_SR = 16000
MIC_SR = 44100

WAKE_DURATION = 1.5     # short listen for wake word
CMD_DURATION = 3        # full command capture


# ---------------- RECORD AUDIO ----------------
def record_audio(duration):

    audio = sd.rec(
        int(duration * MIC_SR),
        samplerate=MIC_SR,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    energy = np.mean(np.abs(audio))
    print("Audio energy:", energy)

    return audio


# ---------------- PREPROCESS ----------------
def preprocess(audio):

    audio = librosa.resample(audio, orig_sr=MIC_SR, target_sr=MODEL_SR)

    audio = librosa.util.normalize(audio)

    audio, _ = librosa.effects.trim(audio, top_db=20)

    TARGET_SAMPLES = MODEL_SR
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    else:
        audio = audio[:TARGET_SAMPLES]

    mfcc = librosa.feature.mfcc(y=audio, sr=MODEL_SR, n_mfcc=40)

    TARGET_LENGTH = 64
    if mfcc.shape[1] < TARGET_LENGTH:
        mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_LENGTH - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :TARGET_LENGTH]

    return mfcc[np.newaxis, ..., np.newaxis]


# ---------------- PREDICT ----------------
def predict(audio):

    X = preprocess(audio)

    prediction = model.predict(X, verbose=0)

    idx = np.argmax(prediction)
    label = encoder.inverse_transform([idx])[0]
    confidence = prediction[0][idx]

    return label, confidence


# ---------------- APP LOOP ----------------
try:
    while True:

        # ===== WAKE WORD LISTENING =====
        audio = record_audio(WAKE_DURATION)
        label, conf = predict(audio)

        if label == "anu" and conf > 0.7:
            print("\n🟢 Wake word detected!")

            print("🎤 Listening for command...")
            time.sleep(0.3)

            # ===== COMMAND LISTENING =====
            cmd_audio = record_audio(CMD_DURATION)
            cmd_label, cmd_conf = predict(cmd_audio)

            print(f"🧠 Command detected: {cmd_label} ({cmd_conf*100:.2f}%)")
            print("🔵 Returning to standby...\n")

except KeyboardInterrupt:
    print("\n🛑 Assistant stopped.")
