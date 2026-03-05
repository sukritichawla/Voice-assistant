import numpy as np
import librosa
import tensorflow as tf
import pickle

print("Loading TensorFlow...")
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model(r"C:\Users\ADMIN\Desktop\Sukriti\VIT\SL\AnuAssistant\scripts\models\best_model.h5")

print("Loading label encoder...")
import pickle
with open(r"C:\Users\ADMIN\Desktop\Sukriti\VIT\SL\AnuAssistant\scripts\models\label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("✅ Model ready!")



# Load label encoder
with open(r"C:\Users\ADMIN\Desktop\Sukriti\VIT\SL\AnuAssistant\scripts\models\label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("✅ Model loaded")

# -------- PREPROCESS FUNCTION --------
def preprocess_audio(audio, sr=16000):

    # normalize (same as training)
    audio = librosa.util.normalize(audio)

    # trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # force 1 second
    TARGET_SAMPLES = 16000
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    else:
        audio = audio[:TARGET_SAMPLES]

    # MFCC extraction
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    TARGET_LENGTH = 64
    if mfcc.shape[1] < TARGET_LENGTH:
        mfcc = np.pad(mfcc, ((0,0),(0,TARGET_LENGTH-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :TARGET_LENGTH]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    return mfcc

# -------- PREDICTION --------
file_path = input("Enter audio file path: ")

# resample mic audio exactly like training loader
audio = librosa.resample(audio, orig_sr=44100, target_sr=16000)

X = preprocess_audio(audio, sr=16000)


prediction = model.predict(X)

predicted_index = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_index])[0]

print("\n🎤 Predicted Command:", predicted_label)
