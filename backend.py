# backend/backend.py
from flask import Flask, request, jsonify
from pymongo import MongoClient
import random

app = Flask(__name__)

# Connect to your MongoDB instance (adjust connection string if necessary)
client = MongoClient("mongodb://localhost:27017")
db = client['makeup_db']
collection = db['makeup_products']

@app.route('/get_reference', methods=['GET'])
def get_reference():
    product_type = request.args.get('product_type')
    shade = request.args.get('shade')
    if not product_type or not shade:
        return jsonify({"error": "Missing parameters"}), 400

    # Query for matching products in the collection
    products = list(collection.find({"product_type": product_type, "shade": shade}))
    if not products:
        return jsonify({"error": "No reference image found"}), 404

    # For demonstration, randomly select one document if multiple exist
    product = random.choice(products)
    return jsonify({"image_url": product["image_url"]})

if __name__ == '__main__':
    # Run the Flask server on port 5000 in debug mode
    app.run(debug=True, port=5000)
