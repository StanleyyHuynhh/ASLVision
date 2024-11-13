import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the processed dataset
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Validate data consistency before training
consistent_data = []
consistent_labels = []
for sample, label in zip(data, labels):
    if len(sample) == 42:  # Ensure each sample has exactly 42 features
        consistent_data.append(sample)
        consistent_labels.append(label)
    else:
        print("Skipped sample with inconsistent feature count.")

# Convert consistent data back to arrays
data = np.array(consistent_data)
labels = np.array(consistent_labels)

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model for later use in inference
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Model saved successfully.")