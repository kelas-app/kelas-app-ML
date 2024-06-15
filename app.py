from flask import Flask, request, jsonify
import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import json
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

# Load environment variables from .env file
load_dotenv()

# Function to fetch data from APIs
def fetch_data(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        data = pd.DataFrame(response.json())
        print(f"Data successfully fetched from {url}")
        return data
    except requests.RequestException as e:
        print(f'Failed to fetch data from {url}: {str(e)}')
        return pd.DataFrame()

# Load JWT token from environment variables
jwt_token = os.getenv('JWT_TOKEN')
headers = {'Authorization': f'Bearer {jwt_token}'}

# API URLs
api_urls = {
    'interactions': 'http://161.97.109.65:3000/api/interactions',
    'users': 'http://161.97.109.65:3000/api/users',
    'products': 'http://161.97.109.65:3000/api/products'
}

# Fetch data from APIs
users = fetch_data(api_urls['users'], headers)
products = fetch_data(api_urls['products'], headers)
interactions = fetch_data(api_urls['interactions'], headers)

# Normalize interactions data if necessary
if 'interactions' in interactions.columns:
    interactions_expanded = pd.json_normalize(interactions['interactions'])
else:
    interactions_expanded = pd.json_normalize(interactions.iloc[:, 0])

# Rename and preprocess columns for collaborative filtering
interactions_expanded['user_id'] = interactions_expanded['userId']
interactions_expanded['product_id'] = interactions_expanded['productId']
interactions_expanded['interaction_value'] = interactions_expanded['interactionValue']

# Encode user_id and product_id for collaborative filtering
user_ids = interactions_expanded['user_id'].unique().tolist()
product_ids = interactions_expanded['product_id'].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
product2product_encoded = {x: i for i, x in enumerate(product_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
productencoded2product = {i: x for i, x in enumerate(product_ids)}

interactions_expanded['user'] = interactions_expanded['user_id'].map(user2user_encoded)
interactions_expanded['product'] = interactions_expanded['product_id'].map(product2product_encoded)

# Load collaborative filtering model and weights
model_path = os.path.join('config', 'kelas_config.json')
weights_path = os.path.join('weights', 'kelas_model.weights.h5')

with open(model_path, 'r') as file:
    model_json = file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(weights_path)

def get_popular_products(top_n=30):
    # Assume 'interactions_expanded' has a column 'interaction_value' indicating the popularity of products
    popular_products = interactions_expanded.groupby('product_id')['interaction_value'].sum().nlargest(top_n).index
    popular_products_df = products[products['_id'].isin(popular_products)]
    return popular_products_df.to_dict(orient='records')

# Function to recommend products for a specific user
def recommend_products(user_id, top_n=30):
    if user_id not in user2user_encoded:
        return get_popular_products(top_n)
    
    user_encoded = user2user_encoded[user_id]
    product_ids = list(product2product_encoded.values())

    user_product_array = np.array([[user_encoded] * len(product_ids), product_ids]).T.astype(int)
    predictions = model.predict([user_product_array[:, 0], user_product_array[:, 1]])
    predictions = predictions.flatten()

    top_indices = predictions.argsort()[-top_n:][::-1]
    recommended_product_ids = [productencoded2product[x] for x in top_indices]

    recommended_products = products[products['_id'].isin(recommended_product_ids)]
    return recommended_products.to_dict(orient='records')

# Function to load Universal Sentence Encoder from TensorFlow Hub
def load_embedding_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to embed text using Universal Sentence Encoder
def embed_text(texts, embed_model):
    embeddings = embed_model(texts)
    return embeddings.numpy()

# Function for semantic search based on cosine similarity
def semantic_search(query, embeddings, texts, embed_model, top_k=10):
    # Embed the query
    query_embedding = embed_model([query]).numpy()
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    
    # Get indices of top k similar products
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Retrieve top k products
    results = products.iloc[top_k_indices]
    return results.to_dict(orient='records')

# Load the semantic model
def load_semantic_model(model_path, weights_path):
    with open(model_path, "r") as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(weights_path)
    return model

def sanitize_data(data):
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_data(value) for key, value in data.items()}
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    else:
        return data

# Flask application setup
app = Flask(__name__)

# Endpoint for collaborative filtering recommendation
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    recommended_products = recommend_products(user_id)
    sanitized_products = sanitize_data(recommended_products)
    return jsonify(sanitized_products)

# Endpoint for semantic search
@app.route('/semantic-search', methods=['GET'])
def semantic_search_endpoint():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter "query" is required'}), 400
    
    # Load Universal Sentence Encoder
    embed = load_embedding_model()
    
    # Prepare text data for embedding
    titles = products['name'].tolist()
    labels = products['category'].tolist()
    combined_text = [f"{label} {title}" for label, title in zip(labels, titles)]
    
    # Generate embeddings for the product descriptions
    embeddings = embed(combined_text)

    # Load semantic model
    model_path = os.path.join('config', 'semanticmodel_config.json')
    weights_path = os.path.join('weights', 'semanticmodel.weights.h5')
    semantic_model = load_semantic_model(model_path, weights_path)
    
    # Perform semantic search
    results = semantic_search(query, embeddings.numpy(), titles, embed)
    return jsonify(results)

# Run the Flask application
if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=8000)
