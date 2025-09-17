import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="AI-Powered PPE Monitoring",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load the YOLOv8 model
# Uses st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Application Title and Description
    st.title("👷 AI-Powered PPE Monitoring")
    st.write(
        "Making PPE Monitoring Easier and More Reliable with AI. "
        "Upload an image to detect Personal Protective Equipment."
    )
    st.write("---")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_path = 'best.pt'  # Make sure this model is in the same folder
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.50, 0.05
    )

    # Load the model
    model = load_model(model_path)

    if model is None:
        st.warning("Please ensure the model file `best.pt` is in the correct directory.")
        return

    # Image uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform detection on the image
        with st.spinner("Detecting PPE..."):
            # The model.predict() function handles all preprocessing
            results = model.predict(image, conf=confidence_threshold)

            # Get the first result object
            result = results[0]

            # Plot the results on the image
            # This returns a BGR numpy array
            annotated_image_bgr = result.plot()
            
            # Convert BGR to RGB for display in Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(annotated_image_rgb, caption="Detection Results", use_column_width=True)

        # Display detection summary
        st.subheader("Detection Summary")
        if len(result.boxes) == 0:
            st.info("No objects were detected with the current confidence threshold.")
        else:
            names = model.names
            detection_counts = {}
            
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = names.get(class_id, "Unknown")
                if class_name in detection_counts:
                    detection_counts[class_name] += 1
                else:
                    detection_counts[class_name] = 1
            
            # Display counts in a more structured way
            for class_name, count in detection_counts.items():
                st.write(f"- **{class_name}**: {count}")

if __name__ == "__main__":
    main()
