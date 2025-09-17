# ==============================================================================
# 1. SETUP: Install necessary libraries
# ==============================================================================
# We use -q (quiet) to keep the output clean.
print("Step 1: Installing libraries...")
!pip install ultralytics pandas seaborn matplotlib -q

import os
import shutil
import requests
import zipfile
import glob
from collections import Counter
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

print("Library installation and imports complete.\n")


# ==============================================================================
# 2. DATA & MODEL PREPARATION: Download images and the pre-trained model
# ==============================================================================
print("Step 2: Downloading dataset and pre-trained YOLOv8 model...")

# --- Dataset Download ---
# Using a public "Hard Hat Detection" dataset from Roboflow Universe as an example.
# NOTE: Replace this URL with the one for your own dataset if you have one.
DATASET_URL = "https://universe.roboflow.com/ds/nZmAmrF32E?key=i8Rk6x8bLq"
DATASET_ZIP_PATH = "ppe_dataset.zip"

response = requests.get(DATASET_URL, stream=True)
with open(DATASET_ZIP_PATH, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

# --- Unzip Dataset ---
DATASET_DIR = "ppe_dataset"
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)  # Clean up previous extractions

with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_DIR)

# --- Model Download ---
# Using a public PPE model from Roboflow Universe.
MODEL_URL = "https://universe.roboflow.com/ds/350nDVOoW1?key=uQzJezxSoh"
MODEL_PT_PATH = "best.pt"

response = requests.get(MODEL_URL, stream=True)
with open(MODEL_PT_PATH, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)
        
print("Dataset and model downloaded and prepared.\n")


# ==============================================================================
# 3. MODEL INFERENCE: Run detection on all images
# ==============================================================================
print("Step 3: Performing inference on the dataset...")

# --- Load the YOLO model ---
model = YOLO(MODEL_PT_PATH)

# --- Define paths ---
# We'll use the 'valid' set from the downloaded data for this example
IMAGES_PATH = os.path.join(DATASET_DIR, 'valid', 'images')
image_files = glob.glob(os.path.join(IMAGES_PATH, '*.jpg'))
RESULTS_DIR = 'detection_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Process images and store detections ---
all_detections = []
print(f"Found {len(image_files)} images to process...")

for img_path in image_files:
    # Perform prediction
    results = model.predict(img_path, conf=0.5, verbose=False) # verbose=False for cleaner output
    
    # Save the annotated image
    result_image = results[0].plot() # .plot() returns a BGR numpy array
    result_image_pil = Image.fromarray(result_image[..., ::-1]) # Convert BGR to RGB
    
    base_filename = os.path.basename(img_path)
    output_path = os.path.join(RESULTS_DIR, base_filename)
    result_image_pil.save(output_path)

    # Store detection data for analysis
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        bbox = box.xyxy[0].tolist() # Bounding box coordinates [x1, y1, x2, y2]
        
        all_detections.append({
            'filename': base_filename,
            'class_name': class_name,
            'confidence': confidence,
            'bbox_x1': bbox[0],
            'bbox_y1': bbox[1],
            'bbox_x2': bbox[2],
            'bbox_y2': bbox[3],
        })

print(f"Inference complete. Annotated images are saved in '{RESULTS_DIR}' folder.\n")


# ==============================================================================
# 4. EDA & ANALYSIS: Analyze the detection results
# ==============================================================================
print("Step 4: Analyzing detection results...")

# --- Create a Pandas DataFrame from the detections ---
if not all_detections:
    print("No detections were made. EDA will be skipped.")
else:
    df = pd.DataFrame(all_detections)
    
    # Save the full detection data to a CSV file
    df.to_csv('detection_summary.csv', index=False)
    print("Full detection data saved to 'detection_summary.csv'.")

    # --- Display Summary ---
    print("\n--- Detection Summary ---")
    print(f"Total Detections: {len(df)}")
    
    class_counts = df['class_name'].value_counts()
    print("\nCounts per Class:")
    print(class_counts)

    # --- Visualize the Class Distribution ---
    plt.figure(figsize=(10, 6))
    sns.countplot(y='class_name', data=df, order=class_counts.index, palette='viridis')
    plt.title('Distribution of Detected PPE Classes', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete.\n")


# ==============================================================================
# 5. VISUAL REVIEW: Display some sample results
# ==============================================================================
print("Step 5: Displaying sample annotated images...")

annotated_images = glob.glob(os.path.join(RESULTS_DIR, '*.jpg'))
sample_images = annotated_images[:4] # Display first 4 samples

if not sample_images:
    print("No annotated images to display.")
else:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten() # Flatten the 2x2 grid to a 1D array
    fig.suptitle('Sample Detection Results', fontsize=20)

    for i, img_path in enumerate(sample_images):
        img = Image.open(img_path)
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path))
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()
    
print("\n--- Notebook execution finished. ---")
print("You can find the output files ('detection_summary.csv' and the 'detection_results' folder) in the file browser on the left.")
