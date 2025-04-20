from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import re
import numpy as np
import joblib
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# ================= Setup ===================
import nltk
nltk.data.path.append("/usr/local/nltk_data")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
app = FastAPI()

# ============ Input schema ============
class PredictRequest(BaseModel):
    text: str
    model: str  # e.g., "ann", "svm", etc.

# ============ Text Cleaning ============
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ============ Model Cache ============
loaded_models = {}

def load_model_components(model_name: str):
    if model_name in loaded_models:
        return loaded_models[model_name]

    try:
        base_path = os.path.join("models", model_name)
        vectorizer_path = os.path.join(base_path, "vectorizer.pkl")
        svd_path = os.path.join(base_path, "svd.pkl")
        encoder_path = os.path.join(base_path, "label_encoder.pkl")

        if model_name == "ann":
            vectorizer = joblib.load(vectorizer_path)
            svd = joblib.load(svd_path)
            encoder = joblib.load(encoder_path)
            model_path = os.path.join(base_path, "model.keras")
            model = tf.keras.models.load_model(model_path)
            loaded_models[model_name] = (model, vectorizer, svd, encoder)
            return model, vectorizer, svd, encoder

        elif model_name in ["svm", "logistic_regression"]:
            model_path = os.path.join(base_path, "model.pkl")
            vectorizer = joblib.load(vectorizer_path)
            svd = joblib.load(svd_path)
            encoder = joblib.load(encoder_path)
            model = joblib.load(model_path)
            loaded_models[model_name] = (model, vectorizer, svd, encoder)
            return model, vectorizer, svd, encoder
        elif model_name == 'linear_regression':
            model_path = os.path.join(base_path, "model.pkl")
            vectorizer = joblib.load(vectorizer_path)
            encoder = joblib.load(encoder_path)
            model = joblib.load(model_path)
            loaded_models[model_name] = (model, vectorizer, encoder)
            return model, vectorizer, encoder
        elif model_name == "naive_bayes":
            model_path = os.path.join(base_path, "naive_bayes_model.pkl")
            nb_data = joblib.load(model_path)
            loaded_models[model_name] = nb_data
            return nb_data

        elif model_name == "knn":
            model_path = os.path.join(base_path, "model.pkl")
            vectorizer = joblib.load(vectorizer_path)
            encoder = joblib.load(encoder_path)
            model = joblib.load(model_path)
            loaded_models[model_name] = (model, vectorizer, encoder)
            return model, vectorizer, encoder

        elif model_name == "decision_tree":
            model_path = os.path.join(base_path, "model.pkl")
            vectorizer = joblib.load(vectorizer_path)
            svd = joblib.load(svd_path)
            encoder = joblib.load(encoder_path)
            model = joblib.load(model_path)
            loaded_models[model_name] = (model, vectorizer, svd, encoder)
            return model, vectorizer,svd,encoder

        elif model_name == "k_means":
            model_path = os.path.join(base_path, "model.pkl")
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
            loaded_models[model_name] = (model, vectorizer)
            return model, vectorizer

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
from sklearn.metrics import pairwise_distances
# ============ Prediction Route ============
@app.post("/predict")
def predict(req: PredictRequest):
    model_name = req.model.lower()

    try:
        cleaned = clean_text(req.text)

        if model_name == "ann":
            model, vectorizer, svd, encoder = load_model_components(model_name)
            vec = vectorizer.transform([cleaned])
            reduced = svd.transform(vec)
            probs = model.predict(reduced)
            pred_idx = np.argmax(probs)
            sentiment = encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[0][pred_idx])
            
        elif model_name == "knn":
            model, vectorizer, encoder = load_model_components(model_name)
            vec = vectorizer.transform([cleaned])
            k = model.n_neighbors
            distances, indices = model.kneighbors(vec, n_neighbors=k)
            nearest_labels = model._y[indices[0]]
            most_common = Counter(nearest_labels).most_common(1)[0]
            sentiment = encoder.inverse_transform([most_common[0]])[0]
            confidence = most_common[1] / k

        elif model_name in ["svm", "logistic_regression"]:
            model, vectorizer, svd, encoder = load_model_components(model_name)
            vec = vectorizer.transform([cleaned])
            reduced = svd.transform(vec)
            pred_idx = model.predict(reduced)[0]
            sentiment = encoder.inverse_transform([int(pred_idx)])[0]
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(reduced)))
            else:
                confidence = 1.0
        elif model_name == 'linear_regression':
            model, vectorizer, encoder = load_model_components(model_name)
            vec = vectorizer.transform([cleaned])
            pred_idx = model.predict(vec)[0]
            sentiment = encoder.inverse_transform([int(pred_idx)])[0]
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(reduced)))
            else:
                confidence = 1.0            
        elif model_name == "naive_bayes":
            nb_data = load_model_components(model_name)
            tokens = cleaned.split()

            log_likelihood_positive = nb_data["log_likelihood_positive"]
            log_likelihood_negative = nb_data["log_likelihood_negative"]
            log_likelihood_neutral = nb_data["log_likelihood_neutral"]
            default_log_prob_pos = nb_data["default_log_prob_pos"]
            default_log_prob_neg = nb_data["default_log_prob_neg"]
            default_log_prob_neu = nb_data["default_log_prob_neu"]
            log_prior_positive = nb_data["log_prior_positive"]
            log_prior_negative = nb_data["log_prior_negative"]
            log_prior_neutral = nb_data["log_prior_neutral"]

            score_pos = log_prior_positive + sum(log_likelihood_positive.get(t, default_log_prob_pos) for t in tokens)
            score_neg = log_prior_negative + sum(log_likelihood_negative.get(t, default_log_prob_neg) for t in tokens)
            score_neu = log_prior_neutral + sum(log_likelihood_neutral.get(t, default_log_prob_neu) for t in tokens)

            scores = {
                "positive": score_pos,
                "negative": score_neg,
                "neutral": score_neu
            }

            sentiment = max(scores, key=scores.get)
            confidence = 1.0  # Optional: you can softmax scores if needed

        elif model_name == "decision_tree":
            model, vectorizer, svd, encoder = load_model_components(model_name)
            vec = vectorizer.transform([cleaned])
            reduced = svd.transform(vec)
            pred_idx = model.predict(reduced)[0]
            sentiment = encoder.inverse_transform([int(pred_idx)])[0]
            confidence = 1.0

        elif model_name == "k_means":
            model, vectorizer= load_model_components(model_name)
            vec = vectorizer.transform([cleaned])
            cluster = int(model.predict(vec)[0])
            if cluster == 1:
                sentiment = 'negative'
            elif cluster == 0:
                sentiment = 'positive'
            else: 
                sentiment = 'neutral'
            confidence = 1.0

        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")

        return {
            "sentiment": str(sentiment),
            "confidence": round(confidence, 4),
            "model_used": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# ============ Preload ============
@app.on_event("startup")
async def preload_models():
    print("🔁 Preloading models into memory...")
    for model_name in [ "linear_regression"]:
        try:
            load_model_components(model_name)
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load model '{model_name}': {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
