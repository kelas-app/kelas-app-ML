import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

load_dotenv()  # Load environment variables from .env file
jwt_token = os.getenv('JWT_TOKEN')

headers = {'Authorization': f'Bearer {jwt_token}'}

api_urls = {
    'interactions': 'http://161.97.109.65:3000/api/interactions',
    'users': 'http://161.97.109.65:3000/api/users',
    'products': 'http://161.97.109.65:3000/api/products'
}

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

# Fetch data from APIs
users = fetch_data(api_urls['users'], headers)
products = fetch_data(api_urls['products'], headers)
interactions = fetch_data(api_urls['interactions'], headers)

# Check if data was fetched successfully
if not users.empty and not products.empty and not interactions.empty:
    print("All data fetched successfully.")
else:
    print("Data fetching failed, check errors and retry.")
    # Optionally, add logic to halt further processing if data is crucial

# Let's assume 'interactions' is a DataFrame with a column containing dictionaries
# First, ensure that the 'interactions' column is appropriately normalized
if 'interactions' in interactions.columns:
    interactions_expanded = pd.json_normalize(interactions['interactions'])
else:
    interactions_expanded = pd.json_normalize(interactions.iloc[:, 0])  # If 'interactions' is the name of DataFrame and not a column

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

# Split the data
train, test = train_test_split(interactions_expanded, test_size=0.2, random_state=42)

# Convert data into required format
x_train = [train['user'].values, train['product'].values]
y_train = train['interaction_value'].values
x_test = [test['user'].values, test['product'].values]
y_test = test['interaction_value'].values

# Load Model
# Update the model and weight file paths
with open("kelas_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("kelas_model_weights.h5")

# Compile loaded model
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Function to get recommendations for a specific user
def recommend_products(user_id, model, interactions, user2user_encoded, product2product_encoded, productencoded2product, products, top_n=30):
    # Check if user_id is in the encoding map
    if user_id not in user2user_encoded:
        print(f"User ID {user_id} not found.")
        return pd.DataFrame()

    user_encoded = user2user_encoded[user_id]
    # Get all encoded product IDs as a list of integers
    product_ids = list(product2product_encoded.values())

    # Create user-product array for prediction
    # Ensure all entries are integers for the model input
    user_product_array = np.array([[user_encoded] * len(product_ids), product_ids]).T.astype(int)

    # Predict interaction values using the model
    predictions = model.predict([user_product_array[:, 0], user_product_array[:, 1]])
    predictions = predictions.flatten()

    # Get top N product indices
    top_indices = predictions.argsort()[-top_n:][::-1]
    # Decode the top indices to product IDs
    recommended_product_ids = [productencoded2product[x] for x in top_indices]
 
    # Filter the products DataFrame to get recommended products using the correct column name
    recommended_products = products[products['_id'].isin(recommended_product_ids)]
    return recommended_products

# Try the model with the specified user ID
user_id = '6665eac87aa0dfec0ad43b2d'
recommended_products = recommend_products(user_id, model, interactions, user2user_encoded, product2product_encoded, productencoded2product, products)
print(f"Recommended products for user {user_id}:")
print(recommended_products)