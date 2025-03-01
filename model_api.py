from flask import Flask, request, send_file, jsonify
import io
from PIL import Image
import torch
from training.config import get_config
from training.inference import Inference

app = Flask(__name__)

# ---------- Model Initialization ----------
# Update the path to your model checkpoint as needed.
checkpoint_path = "ckpts/G.pth"
config = get_config()

# Create a dummy args object to satisfy the Inference class requirements.
class DummyArgs:
    pass

args = DummyArgs()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.save_folder = "dummy_save_folder"
args.name = "dummy_name"

# Load the model once at startup (pre-load to reduce latency)
inference = Inference(config, args, model_path=checkpoint_path)

# ---------- Inference Endpoint ----------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a multipart/form-data POST request with:
      - "source": The user's face image.
      - "reference": The makeup reference image.
    Returns the makeup-transferred image as JPEG.
    """
    if 'source' not in request.files or 'reference' not in request.files:
        return jsonify({'error': 'Both "source" and "reference" images are required.'}), 400

    try:
        source_file = request.files['source']
        reference_file = request.files['reference']
        source_img = Image.open(source_file.stream).convert("RGB")
        reference_img = Image.open(reference_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Error reading images: {str(e)}'}), 400

    try:
        # Run inference using your makeup transfer model
        result_img = inference.transfer(source_img, reference_img, postprocess=True)
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

    # Convert the result image to JPEG and return it
    img_io = io.BytesIO()
    result_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    # For local testing; in production, use a production WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=False)
