import os
from google.cloud import storage

# Constants
BUCKET_NAME = "sentimentann"  # Make sure this matches your actual bucket
MODEL_DIR = "models"

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# GCS file loader
def download_blob(filename: str) -> str:
    """
    Downloads a file from GCS if not already cached locally.

    Args:
        filename (str): GCS path relative to the bucket, e.g. "ann/model.keras"

    Returns:
        str: Local file path
    """
    destination = os.path.join(MODEL_DIR, filename)
    if os.path.exists(destination):
        return destination

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    
    # Ensure parent folders exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    blob.download_to_filename(destination)
    print(f"✅ Downloaded: {filename} → {destination}")
    return destination
