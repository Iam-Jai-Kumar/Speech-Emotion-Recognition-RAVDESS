# Speech Emotion Recognition (SER) using LSTM and RAVDESS

This repository implements a Deep Learning solution for Speech Emotion Recognition (SER). The project focuses on classifying human emotions from audio recordings using the RAVDESS dataset and a hybrid CNN-BiLSTM architecture.

## Project Structure

RAVDESS/: The core dataset containing speech audio files (.wav).
RAVDESS data distribution.ipynb: Exploratory Data Analysis (EDA) notebook focusing on class balance and dataset statistics.
SER 01.ipynb: LSTM Model with MFCC as the only feature extraction technique - 50.56% test accuracy
SER 02.ipynb: Hybrid CNN-BiLSTM Model with Hybrid features (MFCC, Chroma, RMS, Centroid) - 77.78% test accuracy
SER 03.ipynb: Hybrid CNN-BiLSTM Model with Hybrid features (MFCC, Delta-MFCC, Delta-Delta-MFCC, Chroma, RMS, Centroid, Bandwidth, Rolloff, Zero Crossing Rate) - 82.64% test accuracy

## Getting Started

1. Prerequisites
Ensure you have Python 3.x installed. You will need the following libraries:

pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow

2. Dataset
This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
Emotions included: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
Format: 16-bit, 48kHz .wav files.

## Implementation Steps

Step 1: Data Preprocessing & EDA
Visualization: Analyzing waveform patterns and spectrograms for different emotions.
Cleaning: Removing silence and normalizing audio length.
Distribution: Checking the frequency of each emotion label to ensure a balanced training set.

Step 2: Feature Extraction
The most critical step involves converting raw audio into a numerical format the model can understand:
MFCCs (Mel-Frequency Cepstral Coefficients): Extracted using librosa.
Scaling: Using StandardScaler to normalize feature values.
Encoding: Converting categorical emotion labels into numerical vectors using to_categorical.

Step 3: Model Architecture
The system utilizes a sophisticated TensorFlow/Keras pipeline:
Conv1D Layers: To extract spatial/spectral features from the audio signal.
Bidirectional LSTM: To capture the temporal dependencies (how speech changes over time).
Dropout & BatchNormalization: To prevent overfitting and ensure stable training.
Dense Output Layer: Softmax activation for multi-class classification.

Step 4: Training & Optimization
Callbacks: EarlyStopping to halt training when accuracy plateaus and ReduceLROnPlateau to fine-tune the learning rate.
Validation: Splitting data into Training and Testing sets (80/20) to validate performance on unseen data.

Step 5: Evaluation
Performance is assessed through:
Accuracy/Loss Curves: Monitoring training vs. validation.
Confusion Matrix: Identifying which emotions are most frequently confused.
Classification Report: Precision, Recall, and F1-Score for every emotion category.

## Results

The model achieves a test accuracy of 82.64% by effectively combining the feature-extraction power of CNNs with the sequential memory of LSTMs.
