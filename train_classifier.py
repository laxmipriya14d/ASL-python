import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Check the maximum length of data entries
max_length = max(len(entry) for entry in data)

# Pad the sequences to the same length
data_padded = [entry + [0] * (max_length - len(entry)) for entry in data]

# Convert to numpy arrays
data_np = np.asarray(data_padded)
labels_np = np.asarray(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data_np, labels_np, test_size=0.2, shuffle=True, stratify=labels_np)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
