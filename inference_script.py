import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# --- Configuration (Must match your training script's settings) ---
# Ensure these paths are correct relative to where you run this script
MODEL_PATH = 'speech_emotion_dl_model.h5'
LABEL_ENCODER_PATH = 'label_encoder_dl.pkl'

# --- Feature Extraction Function (Must be IDENTICAL to training script) ---
# n_mfcc and max_pad_len should be the same as used in emotion_recognizer_dl.py
def extract_features_dl(file_path, n_mfcc=40, max_pad_len=174):
    """
    Extracts MFCCs, Delta MFCCs, and Delta-Delta MFCCs from an audio file.
    Pads/truncates features to a fixed length.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        combined_features = np.vstack((mfccs, mfccs_delta, mfccs_delta2))

        if combined_features.shape[1] > max_pad_len:
            padded_features = combined_features[:, :max_pad_len]
        else:
            padded_features = np.pad(combined_features, ((0, 0), (0, max_pad_len - combined_features.shape[1])),
                                     mode='constant')

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return padded_features.T # Transpose for (frames, features) shape for Keras

# --- Main Inference Logic ---
if __name__ == "__main__":
    print("Loading pre-trained Speech Emotion Recognition model...")
    try:
        # Load the trained model
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")

        # Load the Label Encoder
        le = joblib.load(LABEL_ENCODER_PATH)
        print(f"Label Encoder loaded successfully from {LABEL_ENCODER_PATH}")

    except Exception as e:
        print(f"Error loading model or label encoder: {e}")
        print("Please ensure 'speech_emotion_dl_model.h5' and 'label_encoder_dl.pkl' are in the same directory.")
        exit()

    # --- Example: Predict emotion for a new audio file ---
    # IMPORTANT: Replace this with the actual path to your new WAV audio file.
    # You can use one of the WAV files from your RAVDESS dataset for a quick test.
    # Example: r"C:\Users\Lenovo\Downloads\RAVDESS\Actor_01\03-01-01-01-01-01-01.wav"
    test_audio_file = r"C:\Users\Lenovo\Downloads\RAVDESS\audio_speech_actors_01-24\Actor_20\03-01-03-01-01-02-20.wav" # <-- CHANGE THIS PATH!

    if not os.path.exists(test_audio_file):
        print(f"\nError: Test audio file not found at {test_audio_file}")
        print("Please change 'test_audio_file' to a valid path to a .wav audio file.")
    else:
        print(f"\nProcessing audio file: {test_audio_file}")
        
        # Extract features from the new audio file
        # Ensure n_mfcc and max_pad_len match what was used during training
        features = extract_features_dl(test_audio_file, n_mfcc=40, max_pad_len=174)

        if features is not None:
            # Reshape features for model prediction: (1, time_steps, features)
            # The model expects a batch dimension (1 for single prediction)
            features = np.expand_dims(features, axis=0)

            # Predict emotion probabilities
            predictions = model.predict(features)

            # Get the predicted class (emotion)
            predicted_class_index = np.argmax(predictions)
            predicted_emotion = le.inverse_transform([predicted_class_index])[0]

            print(f"Predicted Emotion: {predicted_emotion}")
            print(f"Prediction Probabilities: {predictions[0]}")
            print(f"All possible emotions: {le.classes_}")

        else:
            print("Failed to extract features from the audio file.")

    # --- Optional: Add functionality for live microphone input ---
    # This would require additional libraries like 'pyaudio' and more complex setup.
    # For now, focus on file-based prediction.
