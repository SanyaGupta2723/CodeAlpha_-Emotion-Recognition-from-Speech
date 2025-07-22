# emotion_recognizer_dl.py
#  python inference_script.py

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display # Matplotlib's indirect dependency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt # Optional for plotting
import joblib # Added for saving LabelEncoder

# --- Configuration ---
# IMPORTANT: Set the correct path to your RAVDESS dataset here!
DATA_PATH = r"YOUR ACTUAL PATH WHERE THE RAVDESS IS DOWNLOADED" # Example path, adjust if yours is different

# --- Step 1: Feature Extraction Function (MFCCs, Delta, Delta-Delta) ---
def extract_features_dl(file_path, n_mfcc=40, max_pad_len=174):
    """
    Extracts MFCCs, Delta MFCCs, and Delta-Delta MFCCs from an audio file.
    Pads/truncates features to a fixed length.
    n_mfcc: Number of MFCCs to extract.
    max_pad_len: Fixed length to pad/truncate features. This should be determined
                 by the maximum length of features across your dataset during analysis.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Extract Delta MFCCs (first derivative)
        mfccs_delta = librosa.feature.delta(mfccs)

        # Extract Delta-Delta MFCCs (second derivative)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        # Combine all features by stacking them vertically (along the feature dimension)
        combined_features = np.vstack((mfccs, mfccs_delta, mfccs_delta2))

        # Padding/Truncating to a fixed length (crucial for CNN/LSTM input)
        if combined_features.shape[1] > max_pad_len:
            # Truncate if longer than max_pad_len
            padded_features = combined_features[:, :max_pad_len]
        else:
            # Pad if shorter than max_pad_len using constant mode (zeros)
            padded_features = np.pad(combined_features, ((0, 0), (0, max_pad_len - combined_features.shape[1])),
                                     mode='constant')

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
    # Transpose for (frames, features) shape, which is expected by Keras Conv1D/LSTM layers
    return padded_features.T

# --- Step 2: Load Dataset and Extract Features ---
def load_data_dl(data_path):
    """
    Loads audio files from the specified data_path, extracts features,
    and maps filenames to emotion labels.
    """
    data = []
    labels = []
    
    # Mapping emotion codes from RAVDESS filenames to human-readable labels
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }

    # Iterate through actor folders in the dataset path
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if os.path.isdir(actor_path): # Ensure it's a directory
            for filename in os.listdir(actor_path):
                if filename.endswith(".wav"): # Process only WAV files
                    file_path = os.path.join(actor_path, filename)
                    
                    # Extract emotion code from filename (e.g., 03-01-01-01-01-01-01.wav -> code 1)
                    emotion_code = int(filename.split('-')[2])
                    emotion_label = emotion_map.get(emotion_code)
                    
                    if emotion_label: # If a valid emotion label is found
                        features = extract_features_dl(file_path)
                        if features is not None:
                            data.append(features)
                            labels.append(emotion_label)
    
    return np.array(data), np.array(labels)

# --- Step 3: Define Deep Learning Model (CNN + LSTM Example) ---
def create_dl_model(input_shape, num_classes):
    """
    Defines and compiles a Sequential Keras model with Conv1D and LSTM layers.
    input_shape: Tuple (time_steps, features) for the input layer.
    num_classes: Number of output emotion classes.
    """
    model = Sequential([
        # Conv1D layer for local feature extraction in time series data (audio features)
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(), # Normalizes activations
        MaxPooling1D(pool_size=2), # Reduces dimensionality
        Dropout(0.3), # Regularization to prevent overfitting

        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # LSTM layer for capturing long-term dependencies in the sequence
        LSTM(128, return_sequences=False), # return_sequences=False for the last LSTM layer
        Dropout(0.3),

        # Dense layers for classification
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax') # Output layer with softmax for multi-class classification
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Loading data and extracting features for Deep Learning...")
    features, labels = load_data_dl(DATA_PATH)

    if features.size == 0:
        print("No features extracted. Please check DATA_PATH and dataset structure.")
        exit()

    # --- Data Preprocessing for Deep Learning ---
    # Convert string labels to numerical and then one-hot encode
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_one_hot, test_size=0.25, random_state=42, 
        stratify=labels_encoded # Stratify ensures balanced class distribution in splits
    )

    # Determine input shape for the model: (time_steps, features)
    # X_train.shape will be (num_samples, max_pad_len, n_mfcc * 3)
    # So, input_shape is (max_pad_len, n_mfcc * 3)
    input_shape = (X_train.shape[1], X_train.shape[2])

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Input shape for model: {input_shape}")
    print(f"Unique emotions: {le.classes_}")
    print(f"Number of classes: {len(le.classes_)}")

    # --- Step 4: Model Training ---
    print("\nCreating and training the Deep Learning model (CNN + LSTM)...")
    num_classes = len(le.classes_)
    model = create_dl_model(input_shape, num_classes)
    model.summary() # Print model architecture summary

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=50, # Number of training epochs
                        batch_size=32, # Number of samples per gradient update
                        validation_data=(X_test, y_test), # Data for validation during training
                        verbose=1) # Show progress bar during training

    print("Model training complete.")

    # --- Step 5: Model Evaluation ---
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get predictions for classification report
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class indices
    y_true_classes = np.argmax(y_test, axis=1) # Convert one-hot to class indices

    report = classification_report(y_true_classes, y_pred_classes, target_names=le.classes_)
    print("\nClassification Report:")
    print(report)

    # --- Optional: Plotting Training History ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # --- Save the trained model and LabelEncoder ---
    # These files will be saved in the same directory where you run the script
    try:
        model.save('speech_emotion_dl_model.h5')
        print("Deep learning model saved as 'speech_emotion_dl_model.h5'")
    except Exception as e:
        print(f"Error saving model: {e}")

    try:
        joblib.dump(le, 'label_encoder_dl.pkl')
        print("Label Encoder saved as 'label_encoder_dl.pkl'")
    except Exception as e:
        print(f"Error saving label encoder: {e}")

