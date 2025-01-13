import os
import numpy as np
import librosa 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .descriptors import calculate_feature

class Support_Vector_Machine:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.X = np.array([])
        self.y = np.array([])
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.model = None
        self.model_name = None

        self.accuracy = None
        self.confusionMatrix = None
        self.classificationReport = None

    def create_dataset(self, data_path, list_of_descriptors):
        X = []
        y = []
        for person in os.listdir(data_path):
            person_path = os.path.join(data_path, person)
            if os.path.isdir(person_path):
                for word in os.listdir(person_path):
                    word_path = os.path.join(person_path, word)
                    if os.path.isdir(word_path):
                        for audio_file in os.listdir(word_path):
                            if audio_file.endswith(".wav"):
                                audio_path = os.path.join(word_path, audio_file)
                                audio, sr = librosa.load(audio_path, sr=44100, duration=2.0)    
                                
                                # Normalize audio
                                if "AM Index" not in list_of_descriptors:
                                    audio = librosa.util.normalize(audio)
                                
                                features = []
                                for descriptor in list_of_descriptors:
                                    feature = calculate_feature(audio, sr, descriptor)
                                    if isinstance(feature, np.ndarray):
                                        features.extend(feature)
                                    else:
                                        features.append(feature)
                                X.append(features)
                                y.append(f"{word}_{person}")
        self.X = np.array(X)
        self.y = np.array(y)

    def split_dataset(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42, stratify=self.y)

    def train(self):
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.confusionMatrix = confusion_matrix(self.y_test, y_pred)
        self.classificationReport = classification_report(self.y_test, y_pred, zero_division=0)

    def classify(self, audio_path, list_of_descriptors):
        audio, sr = librosa.load(audio_path, sr=44100, duration=2.0)

        # Normalize audio
        if "AM_index" not in list_of_descriptors:
            audio = librosa.util.normalize(audio)
        features = []
        for descriptor in list_of_descriptors:
            feature = calculate_feature(audio, sr, descriptor)
            if isinstance(feature, np.ndarray):
                features.extend(feature)
            else:
                features.append(feature)
        return self.model.predict([features])[0]
    
    def plot_confusion_matrix(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.confusionMatrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(self.y)))
        plt.xticks(tick_marks, set(self.y), rotation=45)
        plt.yticks(tick_marks, set(self.y))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def print_results(self):
        print(f"Accuracy: {self.accuracy}")
        print(f"Confusion Matrix: \n{self.confusionMatrix}")
        print(f"Classification Report: \n{self.classificationReport}")

    def save_model(self, model_name):
        model_path = os.path.join("Models/SVM", model_name)
        import joblib
        joblib.dump(self.model, model_path)

    def load_model(self, model_name):
        model_path = os.path.join("Models/SVM", model_name)
        self.model_name = model_name
        import joblib
        self.model = joblib.load(model_path)

if __name__ == "__main__":

    data_path = "Data"
    list_of_descriptors = ["MFCC"]

    svm = Support_Vector_Machine(kernel='rbf', C=100.0, gamma='scale')
    svm.create_dataset(data_path, list_of_descriptors)
    svm.split_dataset(test_size=0.2)
    svm.train()
    svm.evaluate()
    svm.print_results()
    svm.plot_confusion_matrix()

    audio_path = "Data/Alvaro/Casa/Casa_Alv_1.wav"
    class_ = svm.classify(audio_path, list_of_descriptors)
    print(f"Classified as: {class_}")
