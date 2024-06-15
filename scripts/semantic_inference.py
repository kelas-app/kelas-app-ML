import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import json

load_dotenv()  # Load environment variables from .env file
jwt_token = os.getenv('JWT_TOKEN')

headers = {'Authorization': f'Bearer {jwt_token}'}

api_url = 'http://161.97.109.65:3000/api/products'

# Function to fetch data from API
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

# Fetch data from APIs
products = fetch_data(api_url, headers)

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
    return results

# Load the semantic model
def load_semantic_model(model_path, weights_path):
    with open(model_path, "r") as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(weights_path)
    return model

# Main function for semantic inference
def main():
    # Check if data was fetched successfully
    if products.empty:
        print("Data fetching failed, check errors and retry.")
        return
    
    print("All data fetched successfully.")
    
    # Prepare text data for embedding
    titles = products['name'].tolist()
    labels = products['category'].tolist()
    combined_text = [f"{label} {title}" for label, title in zip(labels, titles)]
    
    # Load the Universal Sentence Encoder
    embed = load_embedding_model()
    
    # Generate embeddings for the product descriptions
    embeddings = embed(combined_text)

    # Convert embeddings to numpy arrays
    embeddings_np = embeddings.numpy()
    
    # Load the semantic model
    model_path = os.path.join('../config', 'semanticmodel_config.json')
    weights_path = os.path.join('../weights', 'semanticmodel.weights.h5')
    semantic_model = load_semantic_model(model_path, weights_path)
    
    # Example of semantic search
    query = "rice cooker miyako"
    results = semantic_search(query, embeddings_np, titles, embed, top_k=10)
    print(results)

# Entry point of the script
if __name__ == "__main__":
    main()
