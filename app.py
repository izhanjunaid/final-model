import streamlit as st
from PIL import Image
import torch
from pymongo import MongoClient

# Import your project modules (the full code you provided earlier)
from training.config import get_config
from training.inference import Inference

# ------------------------------------------------------------------------------
# Helper: Load Inference Model (cached so it isn’t reloaded on every run)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_inference(checkpoint_path: str):
    config = get_config()
    # Create a dummy args object. You can add more attributes as needed.
    class DummyArgs:
        pass
    args = DummyArgs()
    # Use GPU if available; otherwise, fallback to CPU.
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dummy values for required attributes
    args.save_folder = "dummy_save_folder"
    args.name = "dummy_name"
    # Instantiate the inference wrapper with the trained checkpoint
    inference_instance = Inference(config, args, model_path=checkpoint_path)
    return inference_instance

# ------------------------------------------------------------------------------
# Helper: Query MongoDB for product information
# ------------------------------------------------------------------------------
def get_products_by_category(category: str):
    # Connect to your locally installed MongoDB instance (default port 27017)
    client = MongoClient("mongodb://localhost:27017/")
    db = client["makeupDB"]
    collection = db["products"]
    products = list(collection.find({"category": category}))
    client.close()
    return products

# ------------------------------------------------------------------------------
# Streamlit Sidebar: Settings and Reference Image Selection
# ------------------------------------------------------------------------------
st.sidebar.title("Makeup Transfer Settings")

# Choose whether to use the product database or manually upload a reference image
use_product_db = st.sidebar.checkbox("Use Product Database for Reference", value=True)

# Let the user set the checkpoint path (update if needed)
checkpoint_path = st.sidebar.text_input("Checkpoint Path", "ckpts/G.pth")

# Load the inference model (cached)
inference = load_inference(checkpoint_path)

# If using product database, display product selection UI
if use_product_db:
    st.sidebar.subheader("Select Reference Product")
    # Define available product categories (adjust as needed)
    categories = ["Lipstick", "Foundation", "Blush", "Eyemakeup"]
    selected_category = st.sidebar.selectbox("Select Product Category", categories)
    products = get_products_by_category(selected_category)
    if products:
        shade_options = [prod["shade"] for prod in products]
        selected_shade = st.sidebar.selectbox("Select Shade", shade_options)
        ref_item = next((prod for prod in products if prod["shade"] == selected_shade), None)
        if ref_item:
            ref_img_path = ref_item["reference_image_path"]
        else:
            ref_img_path = None
    else:
        st.sidebar.error("No products found for the selected category.")
        ref_img_path = None
else:
    ref_img_path = None
    reference_file = st.sidebar.file_uploader("Upload Reference Image (Makeup)", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------------------------
# Streamlit Main Area: Title and Source Image Upload
# ------------------------------------------------------------------------------
st.title("Virtual Makeup Try-On")
st.write("Upload a source image (without makeup) to try on the selected product.")

source_file = st.file_uploader("Upload Source Image (No Makeup)", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------------------------
# Transfer Makeup Button and Processing
# ------------------------------------------------------------------------------
if st.button("Transfer Makeup"):
    if source_file is None:
        st.error("Please upload a source image!")
    else:
        try:
            source_img = Image.open(source_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading source image: {e}")
            source_img = None

        # Determine the reference image
        if use_product_db:
            if ref_img_path is None:
                st.error("No reference image available for the selected product!")
                reference_img = None
            else:
                try:
                    reference_img = Image.open(ref_img_path).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading reference image from {ref_img_path}: {e}")
                    reference_img = None
        else:
            if reference_file is None:
                st.error("Please upload a reference image for makeup!")
                reference_img = None
            else:
                try:
                    reference_img = Image.open(reference_file).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading uploaded reference image: {e}")
                    reference_img = None

        if source_img is not None and reference_img is not None:
            with st.spinner("Transferring makeup..."):
                # Call the inference transfer method (global mode)
                result_img = inference.transfer(source_img, reference_img, postprocess=True)
            st.image(result_img, caption="Result Image", use_column_width=True)
