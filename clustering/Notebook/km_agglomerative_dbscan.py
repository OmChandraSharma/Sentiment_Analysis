# === SETUP ===
import os
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
# from sklearn.metrics import confusion_matrix
# from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample
# from scipy.stats import mode
from kneed import KneeLocator
import gdown

# === DOWNLOAD & LOAD DATA ===
os.makedirs("plots", exist_ok=True)

file_id = "133jd-yMyIpnVnHPYjiopu0XlpDk6I_H3"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "clean_data.csv", quiet=False)

df = pd.read_csv("clean_data.csv", encoding="utf-8", on_bad_lines="warn")
texts = df['clean_text']
labels = df['sentiment']

label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(labels)
n_clusters = len(np.unique(y_true))

# === VECTORIZERS ===
vectorizers = {
    "TF-IDF": TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    "BoW": CountVectorizer(max_features=5000, ngram_range=(1, 2)),
}

# === HELPERS ===
# def align_clusters(y_true, y_pred):
#     new_labels = np.zeros_like(y_pred)
#     for i in np.unique(y_pred):
#         mask = y_pred == i
#         if np.any(mask):
#             new_labels[mask] = mode(y_true[mask], keepdims=True)[0]
#     return new_labels

# def plot_results(X_vec, y_pred, y_true, method_name, vec_name):
#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=label_encoder.classes_,
#                 yticklabels=label_encoder.classes_)
#     plt.title(f'Confusion Matrix - {method_name} ({vec_name})')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     cm_path = f"plots/confusion_{method_name}_{vec_name}.png"
#     plt.tight_layout()
#     plt.savefig(cm_path)
#     plt.close()
#     print(f" Saved: {cm_path}")

#     # Scatter Plot (PCA using SVD)
#     svd = TruncatedSVD(n_components=2, random_state=42)
#     X_pca = svd.fit_transform(X_vec)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=10)
#     plt.title(f'Scatter - {method_name} ({vec_name})')
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.colorbar()
#     pca_path = f"plots/pca_{method_name}_{vec_name}.png"
#     plt.tight_layout()
#     plt.savefig(pca_path)
#     plt.close()
#     print(f" Saved: {pca_path}")

# === MAIN LOOP ===
for vec_name, vectorizer in vectorizers.items():
    print(f"\n Vectorizing with: {vec_name}")
    X_vec = vectorizer.fit_transform(texts)

    # === KMeans with Elbow Tuning ===
    # print(f" Tuning KMeans for {vec_name}...")
    # sample_size = 10000
    # X_sample = X_vec[:sample_size]
    # y_sample = y_true[:sample_size]

    # inertias = []
    # ks = range(2, 10)
    # for k in ks:
    #     km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    #     km.fit(X_sample)
    #     inertias.append(km.inertia_)

    # elbow_path = f"plots/elbow_kmeans_{vec_name}.png"
    # plt.figure()
    # plt.plot(ks, inertias, marker='o')
    # plt.title(f'Elbow for KMeans - {vec_name}')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Inertia')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(elbow_path)
    # plt.close()
    # print(f" Saved: {elbow_path}")

    # best_k = KneeLocator(ks, inertias, curve='convex', direction='decreasing').knee or 3
    # print(f" Best k: {best_k}")
    best_k=2 #find previously
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X_vec)
    aligned_kmeans = align_clusters(y_true, kmeans_labels)
    # plot_results(X_vec, aligned_kmeans, y_true, "KMeans", vec_name)

    #  Save KMeans model & vectorizer
    with open(f"plots/kmeans_model_{vec_name}.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"plots/vectorizer_{vec_name}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f" Models saved for {vec_name} ")

    # === Agglomerative & DBSCAN on Sampled Dense Data ===
    # print(f" Running Agglomerative & DBSCAN on sample...")
    # X_small, y_small = resample(X_vec, y_true, n_samples=sample_size, random_state=42)
    # svd_50 = TruncatedSVD(n_components=50, random_state=42)
    # X_dense = svd_50.fit_transform(X_small)

    # # Agglomerative Clustering
    # agglo = AgglomerativeClustering(n_clusters=n_clusters)
    # agglo_labels = agglo.fit_predict(X_dense)
    # aligned_agglo = align_clusters(y_small, agglo_labels)
    # plot_results(X_dense, aligned_agglo, y_small, "Agglomerative", vec_name)

    # # DBSCAN
    # dbscan = DBSCAN(eps=1.0, min_samples=5, n_jobs=-1)
    # db_labels = dbscan.fit_predict(X_dense)
    # mask = db_labels != -1
    # if np.sum(mask) > 0:
    #     aligned_db = align_clusters(y_small[mask], db_labels[mask])
    #     plot_results(X_dense[mask], aligned_db, y_small[mask], "DBSCAN", vec_name)
    # else:
    #     print(f" DBSCAN ({vec_name}) found no core points. Try tuning eps/min_samples.")
