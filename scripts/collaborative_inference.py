import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv


# Function to fetch data from APIs
def fetch_data(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        data = pd.DataFrame(response.json())
        print(f"Data successfully fetched from {url}")
        print(data.head())  # Display the first few rows of the DataFrame
        return data
    except requests.RequestException as e:
        print(f'Failed to fetch data from {url}: {str(e)}')
        return pd.DataFrame()

# Load environment variables from .env file
load_dotenv()

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

# Rename and preprocess columns
interactions_expanded['user_id'] = interactions_expanded['userId']
interactions_expanded['product_id'] = interactions_expanded['productId']
interactions_expanded['interaction_value'] = interactions_expanded['interactionValue']

# Encode user_id and product_id
user_ids = interactions_expanded['user_id'].unique().tolist()
product_ids = interactions_expanded['product_id'].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
product2product_encoded = {x: i for i, x in enumerate(product_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
productencoded2product = {i: x for i, x in enumerate(product_ids)}

interactions_expanded['user'] = interactions_expanded['user_id'].map(user2user_encoded)
interactions_expanded['product'] = interactions_expanded['product_id'].map(product2product_encoded)

# Load pre-trained model and weights
model_path = os.path.join('../config', 'kelas_config.json')
weights_path = os.path.join('../weights', 'kelas_model.weights.h5')

with open(model_path, 'r') as file:
    model_json = file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(weights_path)

# Function to recommend products for a specific user
def recommend_products(user_id, model, interactions, user2user_encoded, product2product_encoded, productencoded2product, products, top_n=30):
    if user_id not in user2user_encoded:
        print(f"User ID {user_id} not found.")
        return pd.DataFrame()

    user_encoded = user2user_encoded[user_id]
    product_ids = list(product2product_encoded.values())

    user_product_array = np.array([[user_encoded] * len(product_ids), product_ids]).T.astype(int)
    predictions = model.predict([user_product_array[:, 0], user_product_array[:, 1]])
    predictions = predictions.flatten()

    top_indices = predictions.argsort()[-top_n:][::-1]
    recommended_product_ids = [productencoded2product[x] for x in top_indices]

    recommended_products = products[products['_id'].isin(recommended_product_ids)]
    return recommended_products

# Example usage: Get recommendations for a specific user
if __name__ == "__main__":
    user_id = '6665e9847aa0dfec0ad43b26'
    recommended_products = recommend_products(user_id, model, interactions_expanded, user2user_encoded, product2product_encoded, productencoded2product, products)
    print(f"Recommended products for user {user_id}:")
    print(recommended_products)
