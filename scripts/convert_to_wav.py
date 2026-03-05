from pydub import AudioSegment
import os

DATASET_PATH = "dataset"   # your dataset folder

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.endswith(".wav"):
            file_path = os.path.join(root, file)

            try:
                audio = AudioSegment.from_file(file_path)

                new_path = os.path.splitext(file_path)[0] + ".wav"

                audio.set_frame_rate(16000).set_channels(1).export(
                    new_path, format="wav"
                )

                print("Converted:", file_path)

            except Exception as e:
                print("Error:", file_path, e)

print("✅ Conversion completed!")
