import joblib
import pickle

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


# def load_naive_bayes():
#     try:
#         vectorizer = joblib.load("../naive_bayes/vectorizers/tfidf_vectorizer.pkl")
#         svd = joblib.load("../decision_tree/vectorizers/svd_tfidf.pkl")
#         model = joblib.load("../naive_bayes/model/nb_model_tfidf.pkl")
#         label_encoder = joblib.load("../decision_tree/vectorizers/label_encoder.pkl")
#         return model, vectorizer,svd,label_encoder

#     except Exception as e:
#         print(f"Error loading Naive Bayes model models: {e}")
#         return None, None, None, None
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


# from tensorflow.keras.models import load_model

# def load_ann():
#     try:
#         print("🔁 Loading ANN full model...")

#         # Load vectorizer, SVD, label encoder
#         with open("../ANN/tfidf_vectorizer.pkl", "rb") as f:
#             vectorizer = pickle.load(f)
#         with open("../ANN/svd_vectorizer.pkl", "rb") as f:
#             svd = pickle.load(f)
#         with open("../ANN/label_encoder.pkl", "rb") as f:
#             label_encoder = pickle.load(f)

#         # Load full model
#         model = load_model("../ANN/ann_full_model.keras")  # or .h5 if you saved in HDF5 format
#         print("✅ ANN full model loaded.")

#         return model, vectorizer, svd, label_encoder

#     except Exception as e:
#         print(f"❌ Error loading ANN model: {e}")
#         return None, None, None, None

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
