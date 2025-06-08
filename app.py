# import streamlit as st
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load the pre-trained model
# model = load_model('Invoice_model.h5')  # Ensure the correct model path

# # Define the title and instructions for the app
# st.title("Invoice vs. Non-Invoice Classifier")


# # File uploader for users to upload an image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read and display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert the image to a format suitable for OpenCV processing
#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Preprocess the image as required by the model
#     image = cv2.resize(image, (150, 150))  # Resize to model's input size
#     image = image.reshape((1, 150, 150, 3))  # Reshape for batch processing
#     image = image / 255.0  # Normalize pixel values

#     # Make prediction
#     predictions = model.predict(image)

#     # Display the prediction result
#     if predictions[0][0] < 0.5:
#         st.write("Prediction: **Invoice**")
#     else:
#         st.write("Prediction: **Non-Invoice**")













import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image

# Load the models
@st.cache_resource
def load_models():
    invoice_model = load_model('Invoice_model.h5')
    resnet_model = ResNet50(weights='imagenet')
    return invoice_model, resnet_model

# Define the title and instructions for the app
st.title("Smart Image Classifier")
st.write("First checks if image is an invoice, if not, identifies the actual object!")

try:
    # Load both models
    invoice_model, resnet_model = load_models()
    
    # File uploader for users to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert the image to array
        image_array = np.array(image)
        
        # Process for invoice model
        invoice_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        invoice_image = cv2.resize(invoice_image, (150, 150))
        invoice_image = invoice_image.reshape((1, 150, 150, 3))
        invoice_image = invoice_image / 255.0
        
        # Make invoice prediction
        invoice_predictions = invoice_model.predict(invoice_image, verbose=0)
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification:")
            if invoice_predictions[0][0] < 0.5:
                st.success("✅ This is an Invoice!")
                confidence = (1 - invoice_predictions[0][0]) * 100
                st.write(f"Confidence: {confidence:.2f}%")
            else:
                st.error("❌ This is Not an Invoice!")
                confidence = invoice_predictions[0][0] * 100
                st.write(f"Confidence: {confidence:.2f}%")
                
                # If not an invoice, use ResNet50
                with col2:
                    st.subheader("Sub Classification:")
                    with st.spinner("Analyzing image with ResNet50..."):
                        # Prepare image for ResNet50
                        resnet_image = cv2.resize(image_array, (224, 224))  # ResNet50 requires 224x224
                        resnet_image = np.expand_dims(resnet_image, axis=0)
                        resnet_image = preprocess_input(resnet_image)
                        
                        # Get ResNet50 predictions
                        preds = resnet_model.predict(resnet_image, verbose=0)
                        decoded_preds = decode_predictions(preds, top=3)[0]
                        
                        # Display top 3 predictions
                        st.write("Cateogry:")
                        for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                            label = label.replace('_', ' ').title()
                            st.write(f" {label}: {score*100:.2f}%")
                            
                            # Create a progress bar for confidence
                            st.progress(score)

except Exception as e:
    pass

