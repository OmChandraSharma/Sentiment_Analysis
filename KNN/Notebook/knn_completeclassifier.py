import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("clean_data.csv") 
X = df['clean_text']
y = df['sentiment']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# Define vectorizers
vectorizers = {
    "tfidf": TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
    "bow": CountVectorizer(ngram_range=(1, 2), max_features=5000)
}

# Distance metrics to test
distance_metrics = ["euclidean", "manhattan"]

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("vectors", exist_ok=True)

for vec_name, vectorizer in vectorizers.items():
    # Vectorized data file name
    vec_file = f'vectors/{vec_name}_X_train.pkl'
    
    if os.path.exists(vec_file):
        print(f"Loading cached vectors for {vec_name}")
        with open(vec_file, 'rb') as f:
            X_train_vec, X_test_vec = pickle.load(f)
    else:
        print(f"Vectorizing data using {vec_name}")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        with open(vec_file, 'wb') as f:
            pickle.dump((X_train_vec, X_test_vec), f)

    # Sample 10% for quick k-tuning
    X_sample, _, y_sample, _ = train_test_split(
        X_train_vec, y_train, train_size=0.1, stratify=y_train, random_state=42
    )

    for metric in distance_metrics:
        print(f"\nRunning {vec_name.upper()} + {metric} KNN...\n")
        max_k = 30
        accuracies = []
        k_values = list(range(1, max_k + 1))

        # Use algorithm='auto' which will use KDTree or BallTree depending on the metric and sparsity
        knn_model = KNeighborsClassifier(n_neighbors=max_k, metric=metric, algorithm='auto')
        knn_model.fit(X_sample, y_sample)
        distances, indices = knn_model.kneighbors(X_test_vec, n_neighbors=max_k)

        for k in k_values:
            top_k_indices = indices[:, :k]
            y_pred_k = []

            for neighbors in top_k_indices:
                neighbor_labels = y_sample[neighbors]
                majority_vote = np.bincount(neighbor_labels).argmax()
                y_pred_k.append(majority_vote)

            acc = accuracy_score(y_test, y_pred_k)
            accuracies.append(acc)
            print(f"{vec_name.upper()} + {metric}, k={k}: Accuracy = {acc:.4f}")

        # Save Elbow Plot
        elbow_plot_path = f"plots/elbow_{vec_name}_{metric}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o')
        plt.title(f"Elbow Curve - {vec_name.upper()} + {metric}")
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(elbow_plot_path)
        plt.close()

        # Best k
        best_k = k_values[np.argmax(accuracies)]
        print(f"Best k for {vec_name.upper()} + {metric}: {best_k} with accuracy = {max(accuracies):.4f}")

        # Train full model and save it
        final_knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric, algorithm='auto')
        final_knn.fit(X_train_vec, y_train)

        model_file = f"models/knn_{vec_name}_{metric}_k{best_k}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(final_knn, f)

        # Predict on test
        y_pred = final_knn.predict(X_test_vec)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix - {vec_name.upper()} + {metric}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_plot_path = f"plots/confmat_{vec_name}_{metric}.png"
        plt.savefig(cm_plot_path)
        plt.close()

        print(f"Saved model and plots for {vec_name.upper()} + {metric}\n")

