import tkinter as tk
import numpy as np
import sounddevice as sd
import librosa
import json
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
import tempfile
import os
import webbrowser   # ✅ ADDED
import pyautogui
from datetime import datetime

mode = "wake"   # "wake" or "manual"
# ===== CONFIG =====
MODEL_PATH = "model_2.h5"
LABELS_PATH = "labels_2.json"

SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 20
MAX_PAD_LEN = 40

WAKE_WORD = "anu"

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

# ===== RECORD AUDIO =====
def record_audio():
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    temp_file = tempfile.mktemp(suffix=".wav")
    write(temp_file, SAMPLE_RATE, recording)

    return temp_file

# ===== PREDICT =====
def predict(file_path):
    features = preprocess_audio(file_path)
    prediction = model.predict(features, verbose=0)

    idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return index_to_label[idx], confidence


# ===== COMMAND EXECUTION (✅ ADDED) =====
def execute_command(command):
    command = command.lower().strip()

    print("EXECUTING:", command)  # debug

    # ===== BROWSER =====
    if command == "open browser":
        webbrowser.open("https://www.google.com")

    elif command == "close browser":
        os.system("taskkill /f /im chrome.exe")


    # ===== CALCULATOR =====
    elif command == "open calculator":
        os.system("calc")

    elif command == "close calculator":
        os.system("taskkill /f /im CalculatorApp.exe")

    elif "screenshot" in command:
        try:
            import pyautogui

            pyautogui.FAILSAFE = False

            # 👇 Simulate Print Screen key
            pyautogui.press('printscreen')

            print("📸 Screenshot copied to clipboard")
            result_label.config(text="📸 Screenshot copied (PrtSc)")

        except Exception as e:
            print("❌ Screenshot error:", e)
            result_label.config(text=f"Error: {e}")

    elif "mute" in command or "unmute" in command:
        try:
            import pyautogui
            pyautogui.press("volumemute")

            result_label.config(text="🔇 Muted / Unmuted")

        except Exception as e:
            result_label.config(text=f"Error: {e}")

    elif "play" in command or "pause" in command:
        try:
            import pyautogui
            pyautogui.press("playpause")

            result_label.config(text="⏯️ Play/Pause toggled")

        except Exception as e:
            result_label.config(text=f"Error: {e}")

    elif "next" in command:
        import pyautogui
        pyautogui.press("nexttrack")
        result_label.config(text="⏭️ Next track")
    elif "previous" in command:
        import pyautogui
        pyautogui.press("prevtrack")
        result_label.config(text="⏮️ Previous track")
    elif "help" in command:
        help_text = """
Commands:
- open browser
- close browser
- open calculator
- close calculator
- play / pause
- mute / unmute
- next / previous
- take screenshot
- sleep
"""
        print(help_text)
        result_label.config(text="📜 Help:" \
        """
Commands:
- open browser
- close browser
- open calculator
- close calculator
- play / pause
- mute / unmute
- next / previous
- take screenshot
- sleep
""")
        
    else:
        print("⚠️ Command not mapped:", command)

import time
import numpy as np

def get_energy(audio):
    return np.mean(np.abs(audio))

def listen_loop():
    
    last_trigger_time = 0

    while True:
        global mode

        if mode != "wake":
            time.sleep(0.1)
            continue
        status_label.config(text="🎧 Listening for 'anu'...")
        root.update()

        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()

        audio = recording.flatten()
        energy = get_energy(audio)

        # 🚨 STRICT silence filter
        if energy < 0.02:
            continue

        temp_file = tempfile.mktemp(suffix=".wav")
        write(temp_file, SAMPLE_RATE, recording)

        label, conf = predict(temp_file)
        os.remove(temp_file)

        current_time = time.time()

        # 🚨 Strong cooldown
        if current_time - last_trigger_time < 3:
            continue

        # 🚨 VERY STRICT wake detection
        if label == WAKE_WORD and conf > 0.6:
            last_trigger_time = current_time

            status_label.config(text="✅ Wake word detected!")
            root.update()

            time.sleep(0.2)  # user pause

            # 🎤 Command phase
            status_label.config(text="🎤 Speak command...")
            root.update()

            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()

            audio = recording.flatten()
            energy = get_energy(audio)

            if energy < 0.02:
                result_label.config(text="⚠️ No speech detected")
                continue

            temp_file = tempfile.mktemp(suffix=".wav")
            write(temp_file, SAMPLE_RATE, recording)

            cmd_label, cmd_conf = predict(temp_file)
            os.remove(temp_file)

            # 🚨 Reject bad predictions
            if cmd_label == WAKE_WORD or cmd_conf < 0.55:
                result_label.config(text="⚠️ No valid command")
            else:
                result_label.config(
                    text=f"🧠 Command: {cmd_label}\nConfidence: {cmd_conf:.2f}"
                )

                root.after(0, execute_command, cmd_label)   # ✅ ADDED

            root.update()

            # 🔥 LONG cooldown after command
            time.sleep(3)


def manual_command():
    global mode
    mode = "manual"   # 🚫 stop wake listening

    status_label.config(text="🎤 Speak command...")
    root.update()

    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    audio = recording.flatten()
    energy = get_energy(audio)

    if energy < 0.02:
        result_label.config(text="⚠️ No speech detected")
        mode = "wake"   # ✅ resume listening
        return

    temp_file = tempfile.mktemp(suffix=".wav")
    write(temp_file, SAMPLE_RATE, recording)

    cmd_label, cmd_conf = predict(temp_file)
    os.remove(temp_file)

    print("MANUAL CMD:", cmd_label, cmd_conf)

    if cmd_label == WAKE_WORD or cmd_conf < 0.65:
        result_label.config(text="⚠️ No valid command")
    else:
        result_label.config(
            text=f"🧠 Command: {cmd_label}\nConfidence: {cmd_conf:.2f}"
        )

        root.after(0, execute_command, cmd_label)

    mode = "wake"   # ✅ ALWAYS GO BACK

    
# ===== UI =====
root = tk.Tk()
root.title("Anu Assistant")
root.geometry("400x200")

title = tk.Label(root, text="Anu Voice Assistant", font=("Arial", 16))
title.pack(pady=10)

status_label = tk.Label(root, text="Starting...", font=("Arial", 12))
status_label.pack(pady=10)



result_label = tk.Label(root, text="Command will appear here", font=("Arial", 12))
result_label.pack(pady=10)

btn = tk.Button(root, text="🎤 Give Command", font=("Arial", 12), command=manual_command)
btn.pack(pady=10)

# Run listening in background
import threading
thread = threading.Thread(target=listen_loop, daemon=True)
thread.start()

root.mainloop()