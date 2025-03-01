# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import torch

# Import your inference pipeline
from training.config import get_config
from training.inference import Inference

# --------------------------------------------------------------------------
# Step 1: Load the Inference Model (cached to avoid reloading on every run)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_inference(checkpoint_path: str):
    config = get_config()
    # Create a dummy args object for your Inference class
    class DummyArgs:
        pass
    args = DummyArgs()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.save_folder = "dummy_save_folder"
    args.name = "dummy_name"
    inference_instance = Inference(config, args, model_path=checkpoint_path)
    return inference_instance

# --------------------------------------------------------------------------
# Step 2: Sidebar Setup - Makeup Product Selection and Control Parameters
# --------------------------------------------------------------------------
st.sidebar.title("Makeup Transfer Settings")

# Product type selection
product_type = st.sidebar.selectbox("Select Makeup Product", ["Lipstick", "Eyeshadow", "General"])

# Hardcoded shades for demonstration; ideally, you would query your backend for available shades.
if product_type == "Lipstick":
    shades = ["Ruby Red", "Coral", "Nude"]
elif product_type == "Eyeshadow":
    shades = ["Smokey", "Natural", "Bold"]
else:
    shades = ["Default"]

shade = st.sidebar.selectbox("Select Shade", shades)

# Makeup intensity slider: let users control the strength of the makeup transfer
if product_type in ["Lipstick", "Eyeshadow"]:
    makeup_intensity = st.sidebar.slider("Makeup Intensity", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
else:
    makeup_intensity = st.sidebar.slider("Global Makeup Intensity", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

# Input field for model checkpoint path (ensure your checkpoint is in ckpts/)
checkpoint_path = st.sidebar.text_input("Checkpoint Path", "ckpts/G.pth")

# Load the inference model using the checkpoint path provided
inference = load_inference(checkpoint_path)

# --------------------------------------------------------------------------
# Step 3: Main App - Upload Source Image and Run Makeup Transfer
# --------------------------------------------------------------------------
st.title("Makeup Transfer Website")
st.write("Upload your face image (without makeup) and apply the selected makeup product with adjustable intensity.")

# Upload widget for the user’s face image
source_file = st.file_uploader("Upload Your Face Image", type=["jpg", "jpeg", "png"])

if st.button("Transfer Makeup"):
    if source_file is None:
        st.error("Please upload a face image.")
    else:
        # Open and display the uploaded source image
        source_img = Image.open(source_file).convert("RGB")
        st.image(source_img, caption="Your Uploaded Image", use_column_width=True)
        
        # ----------------------------------------------------------------------
        # Step 4: Retrieve the Reference Image from the Backend API
        # ----------------------------------------------------------------------
        backend_url = "http://localhost:5000/get_reference"
        params = {"product_type": product_type, "shade": shade}
        response = requests.get(backend_url, params=params)
        if response.status_code != 200:
            st.error("Error retrieving reference image: " + response.json().get("error", "Unknown error"))
        else:
            data = response.json()
            ref_url = data["image_url"]
            # Download and display the reference image
            ref_response = requests.get(ref_url)
            if ref_response.status_code != 200:
                st.error("Error loading reference image from the provided URL.")
            else:
                reference_img = Image.open(io.BytesIO(ref_response.content)).convert("RGB")
                st.image(reference_img, caption="Reference Makeup Image", use_column_width=True)
                
                # ------------------------------------------------------------------
                # Step 5: Run the Makeup Transfer Inference Pipeline
                # ------------------------------------------------------------------
                with st.spinner("Transferring makeup..."):
                    if product_type in ["Lipstick", "Eyeshadow"]:
                        # Region-specific transfer:
                        # Preprocess the source and reference images (face detection, mask, landmarks)
                        source_input, face, crop_face = inference.preprocess(source_img)
                        ref_input, _, _ = inference.preprocess(reference_img)
                        if source_input is None or ref_input is None:
                            st.error("Failed to process one or more images. Please try different images.")
                        else:
                            # Generate samples from the preprocessed data; pass makeup_intensity as saturation
                            source_mask = source_input[1]
                            reference_sample = inference.generate_reference_sample(
                                ref_input,
                                source_mask=source_mask,
                                mask_area=product_type.lower(),  # expects 'lipstick' or 'eyeshadow'
                                saturation=makeup_intensity
                            )
                            source_sample = inference.generate_source_sample(source_input)
                            result_img = inference.interface_transfer(source_sample, [reference_sample])
                            result_img = inference.postprocess(source_img, crop_face, result_img)
                    else:
                        # Global makeup transfer:
                        raw_result = inference.transfer(source_img, reference_img, postprocess=True)
                        # Blend the result with the source based on the intensity (here, a simple blend)
                        result_img = Image.blend(source_img, raw_result, makeup_intensity * 0.5)
                
                st.success("Makeup transfer complete!")
                st.image(result_img, caption="Result Image", use_column_width=True)
