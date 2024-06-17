# Kelas Machine Learning

This Flask application provides endpoints for collaborative filtering recommendations and semantic search of products. It integrates with external APIs to fetch user, product, and interaction data and utilizes TensorFlow models for generating recommendations and performing semantic searches.


## Model and Data Handling

- The collaborative filtering model is loaded from a JSON configuration file (`kelas_config.json`) and corresponding weights file (`kelas_model.weights.h5`).
- The Universal Sentence Encoder model is loaded from TensorFlow Hub for embedding text data.
- Data from the external APIs is fetched asynchronously using `aiohttp` and stored in the Flask global context (`g`) for use in request handlers.


## Data Fetching
The application fetches data from the following API URLs:
- Users: `/api/users`
- Products: `/api/products`
- Interactions: `/api/interactions`
Data is sanitized to handle NaN and infinite values, ensuring the API responses are clean and usable.


## Environment Variables

Create a `.env` file in the root directory of your project and add the following environment variables:

`JWT_TOKEN=your_jwt_token_here`

The JWT token used for authorization when fetching data from the APIs.


## Run Locally

Clone the project

```bash
  git clone https://github.com/kelas-app/kelas-app-ml
```

Go to the project directory

```bash
  cd kelas-app-ml
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```
The application will be accessible at http://0.0.0.0:8000.



