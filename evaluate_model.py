import pickle
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
DATA_PATH = './data.pickle'  
with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Load trained model
MODEL_PATH = './model.p'  
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
gesture_model = model_dict['model']

# Make predictions
y_pred = gesture_model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred)

# Measure latency for predictions (in milliseconds)
latency_times = []
for x in x_test:
    start_time = time.time()
    gesture_model.predict([x])
    latency_times.append((time.time() - start_time) * 1000)  # convert to milliseconds

average_latency = np.mean(latency_times)

# Display results
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (weighted): {precision * 100:.2f}%")
print(f"Recall (weighted): {recall * 100:.2f}%")
print(f"F1 Score (weighted): {f1 * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Average Prediction Latency: {average_latency:.2f} ms")
