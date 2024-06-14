import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# ## Load Data
load_dotenv()  # Load environment variables from .env file
jwt_token = os.getenv('JWT_TOKEN')

headers = {'Authorization': f'Bearer {jwt_token}'}

api_urls = {
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
products = fetch_data(api_urls['products'], headers)

# Check if   data was fetched successfully
if not products.empty:
    print("All data fetched successfully.")
else:
    print("Data fetching failed, check errors and retry.")
    # Optionally, add logic to halt further processing if data is crucial

# # Data Preprocessing
# Prepare text data for embedding
titles = products['name'].tolist()
labels = products['category'].tolist()
combined_text = [f"{label} {title}" for label, title in zip(labels, titles)]

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Generate embeddings for the product descriptions
embeddings = embed(combined_text)

# Convert embeddings to numpy arrays
embeddings_np = embeddings.numpy()

# Assuming binary labels for demonstration purposes
labels = (products['category'] == 'Category1').astype(int).values  # Example binary labels based on category

# Split the data
train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(embeddings_np, labels, test_size=0.2, random_state=42)

# Check the shapes of the splits to ensure correctness
print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Validation embeddings shape: {val_embeddings.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Validation labels shape: {val_labels.shape}")

# # Create Model
# Create TensorFlow datasets from the embeddings
def create_tf_dataset(embeddings, labels):
    dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(32)
    return dataset

train_dataset = create_tf_dataset(train_embeddings, train_labels)
val_dataset = create_tf_dataset(val_embeddings, val_labels)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(train_embeddings.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

#semantic function
def semantic_search(query, embed_model, trained_model, embeddings, data, top_k=10):
    # Generate the embedding for the query using the embed_model
    query_embedding = embed_model([query]).numpy()

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Get the top_k products
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    results = products.iloc[top_k_indices]
    return results

# # Try Model
query = "rice cooker miyako"
results = semantic_search(query, embed, model, embeddings_np, products, top_k=10)

print(results)

# Save model configuration and weights
model_json = model.to_json()
with open("semanticmodel_config.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("semanticmodel_weights.h5")

# Load model & weights
with open("semanticmodel_config.json", "r") as json_file:
    loaded_model_json = json_file.read()

model1 = tf.keras.models.model_from_json(loaded_model_json)
model1.load_weights("semanticmodel_weights.h5")

# Compile loaded model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


