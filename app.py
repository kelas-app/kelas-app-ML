import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
jwt_token = os.getenv('JWT_TOKEN')
headers = {'Authorization': f'Bearer {jwt_token}'}

# API URLs
api_urls = {
    'interactions': 'http://161.97.109.65:3000/api/interactions',
    'users': 'http://161.97.109.65:3000/api/users',
    'products': 'http://161.97.109.65:3000/api/products'
}

def fetch_data(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        return pd.DataFrame(response.json())
    except requests.RequestException as e:
        print(f'Failed to fetch data from {url}: {str(e)}')
        return pd.DataFrame()

def preprocess_data():
    # Fetch data from APIs
    users = fetch_data(api_urls['users'], headers)
    products = fetch_data(api_urls['products'], headers)
    interactions = fetch_data(api_urls['interactions'], headers)

    if users.empty or products.empty or interactions.empty:
        print("Data fetching failed, check errors and retry.")
        return None, None, None

    # Normalize the 'interactions' column if it contains dictionaries
    if 'interactions' in interactions.columns:
        interactions_expanded = pd.json_normalize(interactions['interactions'])
    else:
        interactions_expanded = pd.json_normalize(interactions.iloc[:, 0])

    # Assuming the JSON data has keys 'userId', 'productId', and 'interactionValue'
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

    return users, products, interactions_expanded, user2user_encoded, product2product_encoded, productencoded2product

# Preprocess data and load the trained model
users, products, interactions, user2user_encoded, product2product_encoded, productencoded2product = preprocess_data()
model = tf.keras.models.load_model('model/collaborative/collaborative_model.keras')

def recommend_products(user_id, top_n=30):
    # Check if user_id is in the encoding map
    if user_id not in user2user_encoded:
        # If user_id is not in the encoding map, return fallback recommendations
        fallback_recommendations = products.sample(n=top_n)
        return fallback_recommendations[['name', 'category', 'price']].to_dict(orient='records')

    user_encoded = user2user_encoded[user_id]
    # Get all encoded product IDs as a list of integers
    product_ids = list(product2product_encoded.values())

    # Create user-product array for prediction
    user_product_array = np.array([[user_encoded] * len(product_ids), product_ids]).T

    # Predict interaction values using the model
    predictions = model.predict([user_product_array[:, 0], user_product_array[:, 1]])
    predictions = predictions.flatten()

    # Get top N product indices
    top_indices = predictions.argsort()[-top_n:][::-1]
    # Decode the top indices to product IDs
    recommended_product_ids = [productencoded2product[x] for x in top_indices]

    # Filter the products DataFrame to get recommended products
    recommended_products = products[products['_id'].isin(recommended_product_ids)]
    
    return recommended_products[['name', 'category', 'price']].to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    recommendations = recommend_products(user_id)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=8000)
