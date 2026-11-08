ANU REVISION NOTEBOOKS

Run in this order:
1. 01_dataset_audit_and_manifest.ipynb — set dataset folder and inventory WAVs.
2. Fill anu_revision_output/dataset_manifest_TO_FILL.csv with speaker_id, original_file and is_augmented.
3. 02_speaker_split_and_mfcc.ipynb — create speaker-independent split and MFCC arrays.
4. 03_train_evaluate_cnn.ipynb — train/evaluate CNN on held-out speakers.
5. 04_baselines_and_noise_test.ipynb — compare SVM, Random Forest and KNN; run a preliminary synthetic-noise test.
6. 05_end_to_end_metrics.ipynb — fill CSV from actual assistant runs and calculate intent/entity/action metrics.
7. 06_wakeword_latency_metrics.ipynb — fill session CSV from actual wake-word/latency tests.

Install:
pip install numpy pandas librosa soundfile scikit-learn matplotlib joblib tensorflow

Edit only:
- DATASET_DIR in Notebook 01.
- Fill the manifest with correct speaker IDs and augmentation parent links.
- Fill the result CSVs in Notebooks 05 and 06 using actual test observations.

Important:
- Never split augmented derivatives of one original recording across train and test.
- The notebooks use 22,050 Hz, 20 MFCC coefficients and 40 frames.
- Normalization is fitted on training data only.
- End-to-end, wake-word and latency notebooks calculate metrics from real measurements; they do not simulate application success.
- These notebooks do not automatically edit anu_app.py. Its model file, labels, sample rate, MFCC dimensions and normalization must be aligned with the final trained model.
