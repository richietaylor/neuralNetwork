import matplotlib.pyplot as plt
import numpy as np

# Metrics for CustomCNN
customcnn_data = {
    'Accuracy': 0.8184,
    'Precision (Weighted)': 0.8212,
    'Recall (Weighted)': 0.8184,
    'F1-Score (Weighted)': 0.8178
}

# Metrics for RandomForest
randomforest_data = {
    'Accuracy': 0.6329,
    'Precision (Weighted)': 0.6175,
    'Recall (Weighted)': 0.6329,
    'F1-Score (Weighted)': 0.5954
}

# Metric categories
metrics = list(customcnn_data.keys())
customcnn_values = list(customcnn_data.values())
randomforest_values = list(randomforest_data.values())

# Set the positions for the bars
x = np.arange(len(metrics))
width = 0.35  # width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

customcnn_color = '#4682B4'  # Light blue
randomforest_color = '#CD5C5C'  # Light red

# Plot the bars for CustomCNN and RandomForest
rects1 = ax.bar(x - width/2, customcnn_values, width, label='CustomCNN', color=customcnn_color)
rects2 = ax.bar(x + width/2, randomforest_values, width, label='RandomForest', color=randomforest_color)

# Add labels, title, and axis ticks
ax.set_xlabel('Metric Type')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (CustomCNN vs RandomForest)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)  # Set Y-axis limit from 0 to 1
ax.legend()

# Function to add value labels on top of each bar
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',  # Format the value with 4 decimal places
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text above the bar
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels to both sets of bars
add_labels(rects1)
add_labels(rects2)

# Show the plot
plt.tight_layout()
plt.show()
