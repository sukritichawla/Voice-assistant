import tkinter as tk
import threading
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import time
import speech_recognition as sr
import webbrowser

# ---------------- DEVICE ----------------
sd.default.device = 1

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# ---------------- AUDIO SETTINGS ----------------
MODEL_SR = 16000
MIC_SR = 44100
WAKE_DURATION = 1.5
CMD_DURATION = 3


# ================= PREPROCESS =================
def preprocess(audio):
    audio = librosa.resample(audio, orig_sr=MIC_SR, target_sr=MODEL_SR)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    TARGET_SAMPLES = MODEL_SR
    audio = np.pad(audio, (0, max(0, TARGET_SAMPLES-len(audio))))[:TARGET_SAMPLES]

    mfcc = librosa.feature.mfcc(y=audio, sr=MODEL_SR, n_mfcc=40)

    TARGET_LENGTH = 64
    mfcc = np.pad(mfcc, ((0,0),(0,max(0,TARGET_LENGTH-mfcc.shape[1]))))[:, :TARGET_LENGTH]

    return mfcc[np.newaxis, ..., np.newaxis]


def record_audio(duration):
    audio = sd.rec(int(duration * MIC_SR),
                   samplerate=MIC_SR,
                   channels=1,
                   dtype="float32")
    sd.wait()
    return audio.flatten()


def predict(audio):
    X = preprocess(audio)
    pred = model.predict(X, verbose=0)
    idx = np.argmax(pred)
    return encoder.inverse_transform([idx])[0], pred[0][idx]


from difflib import get_close_matches
import speech_recognition as sr
import soundfile as sf



def detect_object_from_audio(audio):

    r = sr.Recognizer()

    # convert float audio → int16 (needed)
    audio_int16 = np.int16(audio * 32767)

    audio_data = sr.AudioData(
        audio_int16.tobytes(),
        sample_rate=44100,
        sample_width=2
    )

    try:
        text = r.recognize_google(audio_data).lower()
        print("Heard:", text)

        keywords = ["browser", "calculator", "screenshot"]

        words = text.split()

        for word in words:

            match = get_close_matches(word, keywords, n=1, cutoff=0.6)

            if match:
                return match[0]

    except:
        print("Could not detect object")

    return None

def execute_command(action, obj=None):

    if action == "open" and obj == "browser":
        webbrowser.open("https://google.com")

    elif action == "open" and obj == "calculator":
        os.system("calc")

    elif action == "close" and obj == "browser":
        os.system("taskkill /im chrome.exe /f")

    elif action == "screenshot":
        os.system("snippingtool")

    elif action == "mute":
        print("Muted")

    elif action == "sleep":
        print("Sleeping mode")
# ================= UI =================
root = tk.Tk()
root.title("Anu Assistant")
root.geometry("420x420")
root.configure(bg="#0f172a")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Anu",
    font=("Segoe UI", 24, "bold"),
    fg="white",
    bg="#0f172a"
)
title.pack(pady=10)

canvas = tk.Canvas(root, width=250, height=250,
                   bg="#0f172a", highlightthickness=0)
canvas.pack()

orb = canvas.create_oval(75, 75, 175, 175,
                         fill="#2563eb", outline="")

status_var = tk.StringVar(value="Listening for wake word...")
status_label = tk.Label(
    root,
    textvariable=status_var,
    font=("Segoe UI", 12),
    fg="white",
    bg="#0f172a"
)
status_label.pack(pady=10)

command_var = tk.StringVar(value="Command: ---")
command_label = tk.Label(
    root,
    textvariable=command_var,
    font=("Segoe UI", 12, "bold"),
    fg="#38bdf8",
    bg="#0f172a"
)
command_label.pack()

running = True
pulse_size = 0
pulse_direction = 1
orb_color = "#2563eb"


# ================= ORB ANIMATION =================
def animate_orb():
    global pulse_size, pulse_direction

    if not running:
        return

    pulse_size += pulse_direction * 1.5

    if pulse_size > 15 or pulse_size < 0:
        pulse_direction *= -1

    canvas.coords(
        orb,
        75 - pulse_size,
        75 - pulse_size,
        175 + pulse_size,
        175 + pulse_size
    )

    canvas.itemconfig(orb, fill=orb_color)

    root.after(40, animate_orb)


# ================= ASSISTANT LOOP =================
def assistant_loop():
    global orb_color

    while running:

        # WAKE MODE
        orb_color = "#2563eb"
        status_var.set("Listening for 'Anu'...")

        audio = record_audio(WAKE_DURATION)
        label, conf = predict(audio)

        if label == "anu" and conf > 0.7:

            orb_color = "#22c55e"
            status_var.set("Wake word detected!")

            time.sleep(0.4)

            # COMMAND MODE
            orb_color = "#a855f7"
            status_var.set("Listening for command...")

            cmd_audio = record_audio(CMD_DURATION)
            cmd_label, cmd_conf = predict(cmd_audio)

            print("ACTION:", cmd_label)

            needs_object = ["open", "close"]

            if cmd_label in needs_object:

                obj = detect_object_from_audio(cmd_audio)

                print("OBJECT:", obj)

                if obj:
                    full_cmd = f"{cmd_label} {obj}"
                else:
                    full_cmd = cmd_label

            else:
                full_cmd = cmd_label
                obj = None

            command_var.set(f"Command: {full_cmd}")

            execute_command(cmd_label, obj)

# ================= CLOSE =================
def on_close():
    global running
    running = False
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

threading.Thread(target=assistant_loop, daemon=True).start()
animate_orb()

root.mainloop()
