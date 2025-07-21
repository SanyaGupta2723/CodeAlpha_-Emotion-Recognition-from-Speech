# 🎧 Speech Emotion Recognition using Deep Learning

📌 Project Overview

This project focuses on recognizing human emotions from speech audio using deep learning and speech signal processing. The primary goal is to classify audio clips into categories like happy, sad, angry, neutral, etc., by analyzing features extracted from audio waveforms.

🎯 Objective

To build a deep learning pipeline that can classify emotions from audio recordings based on extracted speech features like MFCCs (Mel-Frequency Cepstral Coefficients).

⚙️ Approach

Signal Processing: Feature extraction using MFCCs, Delta MFCCs.

Modeling: Deep learning using CNN + LSTM architecture.

Evaluation: Model performance evaluated using accuracy, classification report, and confusion matrix.

📁 Project Structure

.
🕛 emotion_recognizer_dl_final.py        # Main training script
🕛 inference_script_re_provide.py        # Script for emotion prediction from new audio
🕛 speech_emotion_dl_model.h5            # Trained Keras model
🕛 label_encoder_dl.pkl                  # Saved LabelEncoder
🕛 README.md                             # Project documentation
🕛 /dataset/
    └🕛 RAVDESS/                          # Audio dataset directory

🧠 Deep Learning Model

Input: MFCC feature matrix of audio clips

Layers:

Conv1D + MaxPooling1D

LSTM layer

Dense layers with Dropout and BatchNorm

Output: Softmax layer predicting probabilities across emotion labels

🔍 Datasets

RAVDESS

(Optionally) TESS or EMO-DB

🛠️ Key Libraries Used

librosa for audio feature extraction

TensorFlow / Keras for modeling

scikit-learn for preprocessing and evaluation

joblib to save/load encoder

matplotlib and seaborn for visualizations

📊 Results

High classification accuracy on test set

Detailed emotion-wise precision, recall, F1-score

Confusion matrix for visual insight into model predictions

🚀 How to Run

Clone the repo

Prepare RAVDESS dataset inside dataset/

Run training:
```

python emotion_recognizer_dl_final.py
```
Predict using trained model:
```

python inference_script_re_provide.py --file path/to/audio.wav
```

🙌 Acknowledgments

Based on the RAVDESS dataset

Inspired by applications in mental health, emotion AI, and HCI (Human-Computer Interaction)
