import pickle
import requests
import tempfile
# from tensorflow.keras.models import load_model

def download_from_gcs(url, suffix=""):
    """Downloads a file from GCS and returns a temporary file path."""
    response = requests.get(url)
    response.raise_for_status()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(response.content)
    temp.close()
    return temp.name

def load_decision_tree_models():
    try:
        vectorizer = joblib.load("../decision_tree/vectorizers/tfidf_vectorizer.pkl")
        svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")
        model = joblib.load("../decision_tree/models/decision_tree_tf-idf.pkl")
        label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")
        return model, vectorizer, svd, label_encoder
    except Exception as e:
        print(f"Error loading Decision Tree models: {e}")
        return None, None, None, None

def load_naive_bayes():
    try:
        print("Loading Naive Bayes models...")
        vectorizer = joblib.load("../naive_bayes/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../naive_bayes/model/nb_model_tfidf.pkl")  # Check if file exists

        print("Model loaded")

        label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder
    

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None


def load_ann():
    try:
        print("🔁 Loading ANN full model from GCS...")

        # URLs to GCS-hosted files (replace with actual links)
        tfidf_url = "https://storage.googleapis.com/sentimentann/tfidf_vectorizer.pkl"
        svd_url = "https://storage.googleapis.com/sentimentann/svd_vectorizer.pkl"
        label_url = "https://storage.googleapis.com/sentimentann/label_encoder.pkl"
        model_url = "https://storage.googleapis.com/sentimentann/deep_model.weights.h5"

        # Download and load
        with open(download_from_gcs(tfidf_url), "rb") as f:
            vectorizer = pickle.load(f)
        with open(download_from_gcs(svd_url), "rb") as f:
            svd = pickle.load(f)
        with open(download_from_gcs(label_url), "rb") as f:
            label_encoder = pickle.load(f)

        model_path = download_from_gcs(model_url, suffix=".keras")
        model = load_model(model_path)

        print("✅ ANN model and components loaded from GCS.")
        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading ANN model: {e}")
        return None, None, None, None

def load_clusterring():
    try:
        print("Loading Clusterring models...")
        vectorizer = joblib.load("../clusterring/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../clusterring/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../clusterring/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../clusterring/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_KNN():
    try:
        print("Loading KNN models...")
        vectorizer = joblib.load("../knn/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../knn/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../knn/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../knn/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_svm():
    try:
        print("Loading SVM models...")
        vectorizer = joblib.load("../svm/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../svm/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../svm/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../svm/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None

def load_logistic_regression():
    try:
        print("Loading Naive Bayes models...")
        vectorizer = joblib.load("../naive_bayes/vectorisers/tfidf_vectorizer.pkl")
        print("Vectorizer loaded")
        
        svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")  # Confirm if needed
        print("SVD loaded (although unused)")

        model = joblib.load("../naive_bayes/model/nb_model_tfidf.pkl")  # Check if file exists
        print("Model loaded")

        label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")  # Confirm same encoder?
        print("Label encoder loaded")

        return model, vectorizer, svd, label_encoder

    except Exception as e:
        print(f"❌ Error loading Naive Bayes model: {e}")
        return None, None, None, None
