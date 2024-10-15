import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from torchvision import models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set the current working directory
BASE_DIR = Path('.').resolve()  # Current working directory
print(f"Base Directory: {BASE_DIR}")

# Define paths relative to the current working directory
DATA_DIR = BASE_DIR / 'datasets' / 'dataset'

TRAIN_IMAGES_DIR = DATA_DIR / 'images' / 'train'
VAL_IMAGES_DIR = DATA_DIR / 'images' / 'val'
TEST_IMAGES_DIR = DATA_DIR / 'images' / 'test'

TRAIN_LABELS_DIR = DATA_DIR / 'labels' / 'train'
VAL_LABELS_DIR = DATA_DIR / 'labels' / 'val'
TEST_LABELS_DIR = DATA_DIR / 'labels' / 'test'

# Load train and test CSV files from the current working directory
train_csv_path = BASE_DIR / 'Train.csv'
test_csv_path = BASE_DIR / 'Test.csv'
sample_submission_csv_path = BASE_DIR / 'SampleSubmission.csv'

train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
ss = pd.read_csv(sample_submission_csv_path)

def visualize_boxes(image_tensor, bboxes, labels=None, class_names=None, image_name=None):
    """
    Visualize bounding boxes on an image tensor.

    Parameters:
    - image_tensor (torch.Tensor): Image tensor of shape [C, H, W].
    - bboxes (np.array): Array of bounding boxes, with each box in (xmin, ymin, xmax, ymax) format.
    - labels (np.array, optional): Array of class labels corresponding to the bounding boxes.
    - class_names (dict, optional): Mapping of label indices to class names.
    - image_name (str, optional): Name of the image to display as title.
    """
    print(f"Visualizing: {image_name}")
    # Unnormalize the image
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225])  # Multiply by std
    image = image + np.array([0.485, 0.456, 0.406])  # Add mean
    image = np.clip(image, 0, 1)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot each bounding box
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add a label if available
        if labels is not None and class_names is not None:
            label = class_names[labels[i]] if labels[i] in class_names else str(labels[i])
            plt.text(xmin, ymin - 5, label, color='yellow', fontsize=12, weight='bold')

    # Set the title to image name
    if image_name:
        plt.title(image_name, fontsize=14)

    # Display the image with bounding boxes
    plt.axis('off')
    plt.show()

# Add an image_path column
def get_image_path(row):
    image_id = row['Image_ID']
    if os.path.exists(TRAIN_IMAGES_DIR / image_id):
        return TRAIN_IMAGES_DIR / image_id
    elif os.path.exists(VAL_IMAGES_DIR / image_id):
        return VAL_IMAGES_DIR / image_id
    elif os.path.exists(TEST_IMAGES_DIR / image_id):
        return TEST_IMAGES_DIR / image_id
    else:
        raise FileNotFoundError(f"Image {image_id} not found in train, val, or test directories.")

#add image_path column to training
train['image_path'] = train.apply(get_image_path, axis=1)
#add image_path column to testing
test['image_path'] = test.apply(get_image_path, axis=1)

# Map string classes to integer IDs. Encoding.
class_mapper = {x: y for x, y in zip(sorted(train['class'].unique().tolist()), range(train['class'].nunique()))}
train['class_id'] = train['class'].map(class_mapper)

# Drop the 'confidence' column if not needed, which it isnt since the confidence is always 1.
train = train.drop(columns=['confidence'])

# Split data into training and validation sets
print("SPLITTING INTO TRAIN AND VALIDATION")
#drop all duplicate records if they exist.
train_unique_imgs_df = train.drop_duplicates(subset=['Image_ID'], ignore_index=True)
#Split into train and validation sets.
X_train_ids, X_val_ids = train_test_split(
    train_unique_imgs_df['Image_ID'],
    test_size=0.25,
    stratify=train_unique_imgs_df['class'],
    random_state=42
)
#Setting the training dataframes.
X_train = train[train.Image_ID.isin(X_train_ids)]
X_val = train[train.Image_ID.isin(X_val_ids)]

# Define transformations for the images
print("TRANSFORMING IMAGES FOR RESNET")
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize as per ResNet requirements
                         std=[0.229, 0.224, 0.225])
])
# Scale up
def scale_bounding_boxes(bboxes, original_widths, original_heights, resized_width=224, resized_height=224):
    x_scale = original_widths / resized_width
    y_scale = original_heights / resized_height
    scaled_bboxes = bboxes.copy()
    scaled_bboxes[:, 0] = bboxes[:, 0] * x_scale  # xmin
    scaled_bboxes[:, 1] = bboxes[:, 1] * y_scale  # ymin
    scaled_bboxes[:, 2] = bboxes[:, 2] * x_scale  # xmax
    scaled_bboxes[:, 3] = bboxes[:, 3] * y_scale  # ymax
    return scaled_bboxes
#Scale down
def resize_bounding_boxes(bboxes, original_width, original_height, new_width=224, new_height=224):
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    resized_bboxes = bboxes.copy()
    resized_bboxes[:, 0] = bboxes[:, 0] * x_scale  # Scale xmin
    resized_bboxes[:, 1] = bboxes[:, 1] * y_scale  # Scale ymin
    resized_bboxes[:, 2] = bboxes[:, 2] * x_scale  # Scale xmax
    resized_bboxes[:, 3] = bboxes[:, 3] * y_scale  # Scale ymax
    return resized_bboxes

# Define a Custom Dataset Class for Loading Images and Annotations
class CustomDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.image_ids = self.dataframe['Image_ID'].unique()
        self.image_data = self.dataframe.groupby('Image_ID')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.image_data.get_group(image_id)
        image_path = records.iloc[0]['image_path']
        image = Image.open(str(image_path)).convert("RGB")
        original_width, original_height = image.size  # Get original image size

        bboxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
        labels = records['class_id'].values.astype(np.int64)

        # Resize bounding boxes to match the resized image
        bboxes = resize_bounding_boxes(bboxes, original_width, original_height, 224, 224)

        if self.transforms:
            image = self.transforms(image)

        # Return original sizes as well
        return image, bboxes, labels, (original_width, original_height)


#Processing for each batch.
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    bboxes_batch = [item[1] for item in batch]
    labels_batch = [item[2] for item in batch]
    original_sizes = [item[3] for item in batch]  # Collect original sizes
    return images, bboxes_batch, labels_batch, original_sizes

# Create datasets and dataloaders
train_dataset = CustomDataset(dataframe=X_train, transforms=transform)
val_dataset = CustomDataset(dataframe=X_val, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

#### FOR VISUALISATION
# print("VISUALIZING FIRST 5 IMAGES WITH BOUNDING BOXES:")
# for idx in range(5):
#     image, bboxes, labels = val_dataset[idx]
#     image_id = val_dataset.image_ids[idx]
#     visualize_boxes(image, bboxes, labels, class_names=class_mapper, image_name=image_id)
# exit(1)
####

# Load pre-trained ResNet18 and define feature extractor
print("LOADING RESNET PRETRAINED MODEL")
# resnet18 = models.resnet18(pretrained=True)
# resnet18.eval()
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

#Remove the final classification layer, leaving us with image features.
#feature_extractor = nn.Sequential(*list(resnet18.children())[:-1]) 
feature_extractor = nn.Sequential(*list(resnet50.children())[:-1]) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)

# Function to extract features using Resnet
def extract_features(data_loader):
    print("EXTRACTING FEATURES")
    features_list, labels_list, bboxes_list, original_sizes_list = [], [], [], []
    for images, bboxes_batch, labels_batch, original_sizes_batch in tqdm(data_loader):
        images = images.to(device)
        with torch.no_grad():
            # Extract features (everything but last layer of ResNet)
            features = feature_extractor(images)
            # Flatten for usage in random forest
            features = features.view(features.size(0), -1)  # [batch_size, feature_dim]
        for i in range(len(features)):
            num_objects = len(labels_batch[i])
            features_list.extend([features[i].cpu().numpy()] * num_objects)
            labels_list.extend(labels_batch[i])
            bboxes_list.extend(bboxes_batch[i])
            original_sizes_list.extend([original_sizes_batch[i]] * num_objects)
    return np.array(features_list), np.array(labels_list), np.array(bboxes_list), np.array(original_sizes_list)

# Train models and save features
X_train_features, y_train_labels, y_train_bboxes, y_train_sizes = extract_features(train_loader)
X_val_features, y_val_labels, y_val_bboxes, y_val_sizes = extract_features(val_loader)

# Reshape original sizes arrays for broadcasting
y_val_widths = y_val_sizes[:, 0]
y_val_heights = y_val_sizes[:, 1]




print("FITTING FOR CLASSES AND BOXES")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_features, y_train_labels)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_features, y_train_bboxes)

print("PREDICTIONS")
y_val_pred_labels = clf.predict(X_val_features)
y_val_pred_bboxes = reg.predict(X_val_features)

# Scale up predicted bounding boxes
y_val_pred_bboxes_scaled = scale_bounding_boxes(y_val_pred_bboxes, y_val_widths, y_val_heights)

# Scale up ground truth bounding boxes (if needed)
y_val_bboxes_scaled = scale_bounding_boxes(y_val_bboxes, y_val_widths, y_val_heights)

# Calculate and print accuracy and MSE
class_accuracy = accuracy_score(y_val_labels, y_val_pred_labels)
bbox_mse = mean_squared_error(y_val_bboxes_scaled, y_val_pred_bboxes_scaled)
print(f'Validation Classification Accuracy: {class_accuracy:.4f}')
print(f'Validation Bounding Box MSE: {bbox_mse:.4f}')

# Save predictions to CSV
print("SAVING TO CSV")
predictions = []
start_idx = 0  # Initialize a start index for tracking
for images, bboxes_batch, labels_batch, original_sizes_batch in tqdm(val_loader):
    images = images.to(device)
    with torch.no_grad():
        features = feature_extractor(images)
        features = features.view(features.size(0), -1).cpu().numpy()
    
    for idx in range(len(images)):
        image_id = val_dataset.image_ids[start_idx + idx]
        num_objects = len(bboxes_batch[idx])
        
        # Repeat the features for each object
        features_repeated = np.tile(features[idx], (num_objects, 1))
        
        # Predict labels and bounding boxes
        predicted_labels = clf.predict(features_repeated)
        confidences = clf.predict_proba(features_repeated).max(axis=1)
        predicted_bboxes = reg.predict(features_repeated)
        
        # Scale up the predicted bounding boxes
        original_width, original_height = original_sizes_batch[idx]
        predicted_bboxes_scaled = scale_bounding_boxes(
            predicted_bboxes,
            np.array([original_width] * num_objects),
            np.array([original_height] * num_objects)
        )
        
        # Map predicted labels to class names
        class_labels = [
            list(class_mapper.keys())[list(class_mapper.values()).index(pl)]
            for pl in predicted_labels
        ]
        
        for j in range(num_objects):
            confidence = confidences[j]
            xmin, ymin, xmax, ymax = predicted_bboxes_scaled[j]
            predictions.append({
                "Image_ID": image_id,
                "class": class_labels[j],
                "confidence": confidence,
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax
            })
    start_idx += len(images)  # Update the start index

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
