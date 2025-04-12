import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision

# Enable mixed precision
mixed_precision.set_global_policy("mixed_float16")

# Load vectorizers
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer_tfidf = pickle.load(f)
with open('svd_vectorizer.pkl', 'rb') as f:
    svd = pickle.load(f)

# Load LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load and preprocess data
df_all = pd.read_csv('clean_data.csv')
X_tfidf = vectorizer_tfidf.transform(df_all['clean_text'])
X_svd = svd.transform(X_tfidf)

# Convert labels
y = le.transform(df_all['sentiment'])
y_cat = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

# Build ANN
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(256, activation='relu')(input_layer)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
output = Dense(3, activation='softmax', dtype='float32')(x)

deep_model = Model(inputs=input_layer, outputs=output)
deep_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights
deep_model.load_weights("deep_model.weights.h5")

# ===================
# Evaluate ANN model
# ===================
y_test_class = np.argmax(y_test, axis=1)
ann_preds_test = deep_model.predict(X_test)
ann_preds_labels = np.argmax(ann_preds_test, axis=1)

acc = accuracy_score(y_test_class, ann_preds_labels)
print(f"\n[⚡] ANN Accuracy: {acc:.4f}")
print(classification_report(y_test_class, ann_preds_labels, target_names=le.classes_))

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(text: str):
    cleaned = clean_text(text)
    tfidf_vec = vectorizer_tfidf.transform([cleaned])
    svd_vec = svd.transform(tfidf_vec)

    pred_probs = deep_model.predict(svd_vec)
    pred_class = np.argmax(pred_probs, axis=1)[0]

    sentiment = le.inverse_transform([pred_class])[0]
    confidence = pred_probs[0][pred_class]

    print(f"\n[🔮] Prediction: {sentiment} ({confidence*100:.2f}% confidence)")

# Example usage
predict_sentiment("Wow I really hated the service and the food was awful.")
