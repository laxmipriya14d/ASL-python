
import numpy as np
import matplotlib.pyplot as plt



# Metrics for the larger dataset
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-score']
rfc_values = [91.27, 92.31, 90.91, 91.67, 90.93]
cnn_values = [87.30, 88.46, 87.12, 87.50, 87.79]

# Create a bar plot for accuracy comparison
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, rfc_values, width, label='Random Forest')
bars2 = ax.bar(x + width/2, cnn_values, width, label='CNN')

ax.set_xlabel('Metrics')
ax.set_ylabel('Percentage')
ax.set_title('Performance Comparison: Random Forest vs. CNN (Dataset Size: 2520)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add value labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

fig.tight_layout()
plt.show()




# Data
models = ['Random Forest', 'CNN']
training_accuracy = [89, 95]
testing_accuracy = [98.5, 97]

# Plotting
fig, ax = plt.subplots()
ax.bar(models, training_accuracy, label='Training Accuracy')
ax.bar(models, testing_accuracy, bottom=training_accuracy, label='Testing Accuracy')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Comparison')
ax.legend()

plt.show()

# Data
precision = [0.98, 0.96]  # Hypothetical
recall = [0.985, 0.95]  # Hypothetical
f1_score = [0.9825, 0.955]  # Hypothetical

# Plotting
bar_width = 0.25
index = np.arange(len(models))

fig, ax = plt.subplots()
bar1 = ax.bar(index, precision, bar_width, label='Precision')
bar2 = ax.bar(index + bar_width, recall, bar_width, label='Recall')
bar3 = ax.bar(index + 2 * bar_width, f1_score, bar_width, label='F1-score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, F1-score Comparison')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(models)
ax.legend()

plt.show()
# Data
training_time = [60, 120]  # in minutes (hypothetical for CNN)
inference_time = [30, 50]  # in ms per image (hypothetical for CNN)

# Plotting
fig, ax = plt.subplots()
bar1 = ax.bar(index, training_time, bar_width, label='Training Time (mins)')
bar2 = ax.bar(index + bar_width, inference_time, bar_width, label='Inference Time (ms)')

ax.set_xlabel('Models')
ax.set_ylabel('Time')
ax.set_title('Training and Inference Time Comparison')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

plt.show()
import matplotlib.pyplot as plt

# Data
models = ['Random Forest', 'CNN']
training_accuracy = [89, 95]
testing_accuracy = [98.5, 97]
training_time = [60, 120]  # in minutes

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(training_time, training_accuracy, color='blue', label='Training Accuracy')
plt.scatter(training_time, testing_accuracy, color='red', label='Testing Accuracy')

for i, model in enumerate(models):
    plt.annotate(model, (training_time[i], training_accuracy[i]), textcoords="offset points", xytext=(5,-10), ha='center')
    plt.annotate(model, (training_time[i], testing_accuracy[i]), textcoords="offset points", xytext=(5,10), ha='center')

plt.xlabel('Training Time (minutes)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Training Time')
plt.legend()
plt.grid(True)
plt.show()
# Data
f1_score = [0.9825, 0.955]
inference_time = [30, 50]  # in ms per image

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(inference_time, f1_score, color='purple')

for i, model in enumerate(models):
    plt.annotate(model, (inference_time[i], f1_score[i]), textcoords="offset points", xytext=(5,-10), ha='center')

plt.xlabel('Inference Time (ms)')
plt.ylabel('F1-score')
plt.title('F1-score vs. Inference Time')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Hypothetical data for learning curves
epochs = np.arange(1, 21)
rf_training_accuracy = np.linspace(0.6, 0.89, 20)
rf_validation_accuracy = np.linspace(0.65, 0.85, 20)
cnn_training_accuracy = np.linspace(0.5, 0.95, 20)
cnn_validation_accuracy = np.linspace(0.55, 0.97, 20)

# Plotting Learning Curves
plt.figure(figsize=(14, 6))

# Random Forest Learning Curve
plt.subplot(1, 2, 1)
plt.plot(epochs, rf_training_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs, rf_validation_accuracy, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Random Forest Learning Curve')
plt.legend()
plt.grid(True)

# CNN Learning Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, cnn_training_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs, cnn_validation_accuracy, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Learning Curve')
plt.legend()
plt.grid(True)

plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Hypothetical confusion matrix data
# For illustration purposes, let's assume binary classification (ASL gesture 1 vs ASL gesture 2)
rf_confusion_matrix = np.array([[45, 5], [7, 43]])
cnn_confusion_matrix = np.array([[48, 2], [4, 46]])

# Plotting Confusion Matrices
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest Confusion Matrix
sns.heatmap(rf_confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=axs[0])
axs[0].set_title('Random Forest Confusion Matrix')
axs[0].set_xlabel('Predicted')
axs[0].set_ylabel('Actual')

# CNN Confusion Matrix
sns.heatmap(cnn_confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=axs[1])
axs[1].set_title('CNN Confusion Matrix')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('Actual')

plt.show()
from sklearn.metrics import roc_curve, auc

# Hypothetical ROC data
rf_fpr = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
rf_tpr = np.array([0.0, 0.7, 0.8, 0.9, 1.0])
cnn_fpr = np.array([0.0, 0.05, 0.1, 0.2, 1.0])
cnn_tpr = np.array([0.0, 0.75, 0.85, 0.95, 1.0])

# Calculate AUC
rf_auc = auc(rf_fpr, rf_tpr)
cnn_auc = auc(cnn_fpr, cnn_tpr)

# Plotting ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, color='blue', label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(cnn_fpr, cnn_tpr, color='red', label=f'CNN (AUC = {cnn_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
from sklearn.metrics import precision_recall_curve

# Hypothetical precision-recall data
rf_precision = np.array([0.8, 0.82, 0.85, 0.87, 0.89])
rf_recall = np.array([0.6, 0.65, 0.7, 0.75, 0.8])
cnn_precision = np.array([0.85, 0.88, 0.9, 0.92, 0.95])
cnn_recall = np.array([0.65, 0.7, 0.75, 0.8, 0.85])

# Plotting Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(rf_recall, rf_precision, color='blue', label='Random Forest')
plt.plot(cnn_recall, cnn_precision, color='red', label='CNN')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
