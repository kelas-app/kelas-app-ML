# Use an official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and other necessary files
COPY app.py .
COPY requirements.txt .
COPY config ./config
COPY scripts ./scripts
COPY weights ./weights
COPY .env .

# Install the dependencies including gunicorn
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Expose the port on which the Gunicorn server will run
EXPOSE 8000

# Start the Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]