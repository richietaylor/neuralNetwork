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

# Add an image_path column
# Adjusted to include 'train', 'val', or 'test' directories based on 'Image_ID'
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

train['image_path'] = train.apply(get_image_path, axis=1)
test['image_path'] = test.apply(get_image_path, axis=1)

# Map string classes to integer IDs. Encoding.
class_mapper = {x: y for x, y in zip(sorted(train['class'].unique().tolist()), range(train['class'].nunique()))}
train['class_id'] = train['class'].map(class_mapper)

# Drop the 'confidence' column if not needed
# Alternatively, if you need to use 'confidence', adjust the code accordingly
# For this implementation, we'll ignore 'confidence'
train = train.drop(columns=['confidence'])

# Split data into training and validation sets
train_unique_imgs_df = train.drop_duplicates(subset=['Image_ID'], ignore_index=True)
X_train_ids, X_val_ids = train_test_split(
    train_unique_imgs_df['Image_ID'],
    test_size=0.25,
    stratify=train_unique_imgs_df['class'],
    random_state=42
)

X_train = train[train.Image_ID.isin(X_train_ids)]
X_val = train[train.Image_ID.isin(X_val_ids)]

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 as required by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize as per ResNet requirements
                         std=[0.229, 0.224, 0.225])
])

def resize_bounding_boxes(bboxes, original_width, original_height, new_width=224, new_height=224):
    """
    Resize bounding boxes according to the scaling factor of the image resize.
    
    Parameters:
    - bboxes (np.array): Array of bounding boxes with shape (N, 4), where N is the number of boxes, 
                         and each box is represented as (xmin, ymin, xmax, ymax).
    - original_width (int): Original width of the image.
    - original_height (int): Original height of the image.
    - new_width (int): The width to which the image is resized (default: 224).
    - new_height (int): The height to which the image is resized (default: 224).
    
    Returns:
    - resized_bboxes (np.array): Array of resized bounding boxes with shape (N, 4).
    """
    
    # Calculate scale factors for both dimensions
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    
    # Apply scaling to each bounding box coordinate
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

        # Group data by images
        self.image_ids = self.dataframe['Image_ID'].unique()
        self.image_data = self.dataframe.groupby('Image_ID')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.image_data.get_group(image_id)
        image_path = records.iloc[0]['image_path']
        
        #Open image & get original dimensions
        image = Image.open(str(image_path)).convert("RGB")
        original_width, original_height = image.size

        if self.transforms:
            image = self.transforms(image)

        # Get all bounding boxes and labels for this image
        bboxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
        #Resize the bounding boxes
        bboxes = resize_bounding_boxes(bboxes, original_width, original_height, 224, 224)
        
        labels = records['class_id'].values.astype(np.int64)

        return image, bboxes, labels

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    bboxes_batch = [item[1] for item in batch]
    labels_batch = [item[2] for item in batch]
    return images, bboxes_batch, labels_batch

# Create datasets
train_dataset = CustomDataset(dataframe=X_train, transforms=transform)
val_dataset = CustomDataset(dataframe=X_val, transforms=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

# Load pre-trained ResNet18 and define feature extractor
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])  # Remove the final classification layer

# Move feature extractor to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)

# Function to extract features
def extract_features(data_loader):
    features_list = []
    labels_list = []
    bboxes_list = []
    for images, bboxes_batch, labels_batch in tqdm(data_loader):
        # Move data to device
        images = images.to(device)
        # Pass images through feature extractor
        with torch.no_grad():
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # Flatten to [batch_size, 512]

        # Since some images may have multiple bounding boxes and labels, replicate features accordingly
        for i in range(len(features)):
            num_objects = len(labels_batch[i])
            # Replicate features for each object in the image
            features_list.extend([features[i].cpu().numpy()] * num_objects)
            labels_list.extend(labels_batch[i])
            bboxes_list.extend(bboxes_batch[i])

    X_features = np.array(features_list)
    y_labels = np.array(labels_list)
    y_bboxes = np.array(bboxes_list)

    return X_features, y_labels, y_bboxes

# Extract features for training data
print("Extracting features from training data...")
X_train_features, y_train_labels, y_train_bboxes = extract_features(train_loader)

with open("x_train_features.txt","w") as file:
    file.write(str(X_train_features))

with open("y_train_labels.txt","w") as file:
    file.write(str(y_train_labels))

# Extract features for validation data
print("Extracting features from validation data...")
X_val_features, y_val_labels, y_val_bboxes = extract_features(val_loader)

with open("x_val_features.txt","w") as file:
    file.write(str(X_val_features))

with open("y_val_labels.txt","w") as file:
    file.write(str(y_val_labels))

# Train Random Forest Classifier for classification
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_features, y_train_labels)

# Predict on validation set
y_val_pred_labels = clf.predict(X_val_features)
class_accuracy = accuracy_score(y_val_labels, y_val_pred_labels)
# This tests to see if the classification is correct, disregarding boundary boxes
print(f'Validation Classification Accuracy: {class_accuracy:.4f}')

# Train Random Forest Regressor for bounding box prediction
print("Training Random Forest Regressor...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_features, y_train_bboxes)

# Predict bounding boxes on validation set
y_val_pred_bboxes = reg.predict(X_val_features)
bbox_mse = mean_squared_error(y_val_bboxes, y_val_pred_bboxes)
# This tests to see if the boundary boxes are accurate.
print(f'Validation Bounding Box MSE: {bbox_mse:.4f}')
