import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import os

# Define class names for better visualization
activity_classes = ['Hand Up', 'Hand Down', 'Hand Left', 'Hand Right', 'Hand Circle', 'Hand Cross']
location_classes = [f'Location {i}' for i in range(16)]

# Load test data ground truth
print("Loading test data...")
test_data = sio.loadmat('data/test_data_split_amp.mat')
test_activity_label = test_data['test_activity_label'].squeeze()
test_location_label = test_data['test_location_label'].squeeze()

# Load predictions
print("Loading predictions...")
act_predictions = sio.loadmat('vis/actResult.mat')['act_prediction'].squeeze()
loc_predictions = sio.loadmat('vis/locResult.mat')['loc_prediction'].squeeze()

# Create output directory for visualizations
os.makedirs('vis/figures', exist_ok=True)

# Calculate metrics
act_accuracy = accuracy_score(test_activity_label, act_predictions)
loc_accuracy = accuracy_score(test_location_label, loc_predictions)

print(f"Activity Recognition Accuracy: {act_accuracy*100:.2f}%")
print(f"Location Recognition Accuracy: {loc_accuracy*100:.2f}%")

# Generate confusion matrices
act_cm = confusion_matrix(test_activity_label, act_predictions)
loc_cm = confusion_matrix(test_location_label, loc_predictions)

# Normalize confusion matrices for better visualization
act_cm_norm = act_cm.astype('float') / act_cm.sum(axis=1)[:, np.newaxis]
loc_cm_norm = loc_cm.astype('float') / loc_cm.sum(axis=1)[:, np.newaxis]

# Create classification reports
act_report = classification_report(test_activity_label, act_predictions, 
                                  target_names=activity_classes, output_dict=True)
loc_report = classification_report(test_location_label, loc_predictions, 
                                 target_names=location_classes, output_dict=True)

# Extract precision, recall, and f1-score for visualization
act_metrics = {
    'precision': [act_report[cls]['precision'] for cls in activity_classes],
    'recall': [act_report[cls]['recall'] for cls in activity_classes],
    'f1-score': [act_report[cls]['f1-score'] for cls in activity_classes]
}

loc_metrics = {
    'precision': [loc_report[cls]['precision'] for cls in location_classes],
    'recall': [loc_report[cls]['recall'] for cls in location_classes],
    'f1-score': [loc_report[cls]['f1-score'] for cls in location_classes]
}

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title, filename, normalize=False):
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = cm.max()
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=vmax)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to plot metrics
def plot_metrics(metrics, classes, title, filename):
    n_classes = len(classes)
    n_metrics = len(metrics)
    
    x = np.arange(n_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * (i - n_metrics/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=metric_name)
    
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Activity Recognition Results
print("Generating activity recognition visualizations...")
plot_confusion_matrix(act_cm, activity_classes, 
                     'Activity Recognition Confusion Matrix', 
                     'vis/figures/activity_confusion_matrix.png')

plot_confusion_matrix(act_cm_norm, activity_classes, 
                     'Activity Recognition Normalized Confusion Matrix', 
                     'vis/figures/activity_confusion_matrix_norm.png', 
                     normalize=True)

plot_metrics(act_metrics, activity_classes, 
            'Activity Recognition Metrics', 
            'vis/figures/activity_metrics.png')

# Plot Location Recognition Results
print("Generating location recognition visualizations...")
plot_confusion_matrix(loc_cm, location_classes, 
                     'Location Recognition Confusion Matrix', 
                     'vis/figures/location_confusion_matrix.png')

plot_confusion_matrix(loc_cm_norm, location_classes, 
                     'Location Recognition Normalized Confusion Matrix', 
                     'vis/figures/location_confusion_matrix_norm.png', 
                     normalize=True)

plot_metrics(loc_metrics, location_classes, 
            'Location Recognition Metrics', 
            'vis/figures/location_metrics.png')

# Plot overall comparison
print("Generating overall performance comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

tasks = ['Activity Recognition', 'Location Recognition']
accuracies = [act_accuracy, loc_accuracy]
colors = ['#3498db', '#2ecc71']

bars = ax.bar(tasks, accuracies, color=colors)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}',
               xy=(bar.get_x() + bar.get_width() / 2, height),
               xytext=(0, 3),
               textcoords="offset points",
               ha='center', va='bottom', fontsize=14)

ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy', fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.savefig('vis/figures/overall_performance.png')
plt.close()

print(f"Visualizations saved to 'vis/figures/' directory")
print("\nDetailed metrics:")
print("\nActivity Recognition Classification Report:")
print(classification_report(test_activity_label, act_predictions, target_names=activity_classes))
print("\nLocation Recognition Classification Report:")
print(classification_report(test_location_label, loc_predictions, target_names=location_classes))

