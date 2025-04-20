# Sentiment Analysis Using Classical Machine Learning

**Live App**: [Streamlit Demo](https://omchandrasharma-sentiment-analysis-sentiment-appapp-p762tr.streamlit.app/)  
**Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)  
**Demo Video**: [Watch on YouTube](https://www.youtube.com/watch?v=KabCzsSiWME)

---

## Project Overview

This project presents a unified sentiment analysis platform built using classical machine learning models. The primary goal is to classify textual data (e.g., reviews or comments) based on emotional tone — positive or negative.

### Features

- Text preprocessing (cleaning, normalization)
- TF-IDF and Bag-of-Words (BoW) vectorization
- Dimensionality reduction using Singular Value Decomposition (SVD)
- Multiple ML algorithms:
  - Logistic Regression
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Decision Tree
  - Artificial Neural Network (ANN)
  - Support Vector Machine (SVM)
  - Clustering (Unsupervised)
- Interactive real-time inference and model comparison via a Streamlit dashboard
- Modular and scalable architecture

---

## How to Use the Web App

1. Open the app:  
   [https://omchandrasharma-sentiment-analysis-sentiment-appapp-p762tr.streamlit.app/](https://omchandrasharma-sentiment-analysis-sentiment-appapp-p762tr.streamlit.app/)

2. In the left sidebar, select any model (e.g., Logistic Regression, KNN, Naive Bayes)

3. Scroll to the "Live Simulation" section

4. Enter a sentence or review in the input box

5. Click the "Predict" button

6. View the output label (Positive/Negative) and the prediction confidence score

---

## Folder Structure

.
├── ANN/ # Artificial Neural Network model, graphs & analysis
├── backend/ # FastAPI backend for model inference
├── clustering/ # Clustering algorithms and visualizations
├── decision_tree/ # Decision Tree implementation and analysis
├── KNN/ # K-Nearest Neighbors model and plots
├── linear_regression/ # Linear Regression model, graphs & performance
├── logistic_regression/ # Logistic Regression implementation & visuals
├── naive_bayes/ # Naive Bayes model for classification & sentiment
├── sentiment_app/ # Sentiment analysis logic and app code
├── svm/ # Support Vector Machine implementation and evaluation
├── .DS_Store
├── .gitattributes
├── README.md
├── Sentiment Analysis Dataset Report.pdf
├── SentimentClassificationReport.pdf

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
