import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        # Calculate STFT for chroma feature extraction
        if chroma or mel:
            stft = np.abs(librosa.stft(X))

        # Extract MFCC features
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        # Extract Chroma features
        if chroma:
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feature))

        # Extract Mel Spectrogram features
        if mel:
            mel_feature = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feature))

        return result


# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("C:\\Users\\hegde\\OneDrive\\Desktop\\Speech Emotion Recognition\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue

        try:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            if feature.size > 0:  # Check if features were extracted
                x.append(feature)
                y.append(emotion)
        except Exception as e:
            print(f"Error extracting features from {file_name}: {e}")

    # Ensure we don't pass an empty dataset
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No data was loaded. Check the file paths or feature extraction.")

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Split the dataset
try:
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    print(f'Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}')
except ValueError as e:
    print(f"Data loading error: {e}")

# Check if features were extracted
if x_train.shape[0] > 0 and x_train.shape[1] > 0:
    print(f'Features extracted: {x_train.shape[1]}')
else:
    print("Error: Feature extraction failed or data is empty.")

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
try:
    model.fit(x_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

# Predict for the test set
try:
    y_pred = model.predict(x_test)
except Exception as e:
    print(f"Error during prediction: {e}")

# Calculate the accuracy of our model
try:
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
except Exception as e:
    print(f"Error calculating accuracy: {e}")


# Function to predict emotion from an individual audio file
def predict_emotion(file_name):
    try:
        # Extract features for the individual file
        individual_feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True)

        # Reshape the feature to match the model input (1 sample, N features)
        individual_feature = individual_feature.reshape(1, -1)

        # Predict the emotion
        predicted_emotion = model.predict(individual_feature)

        # Output the predicted emotion
        print(f'Predicted emotion for {file_name}: {predicted_emotion[0]}')

    except Exception as e:
        print(f"Error predicting emotion: {e}")


# Example: Predict emotion for an individual file
predict_emotion("C:\\Users\\hegde\\OneDrive\\Desktop\\Speech Emotion Recognition\\speech-emotion-recognition-ravdess-data\\Actor_01\\03-01-05-02-02-02-01.wav")
