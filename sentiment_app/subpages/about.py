import streamlit as st

def render():
    # st.set_page_config(page_title="About - Sentiment Analysis App", layout="wide")

    st.title("About This App")
    st.markdown("---")

    st.markdown("""
    ## Project Overview
    This Sentiment Analysis Dashboard is a comprehensive platform designed to demonstrate the capabilities of classical machine learning models in detecting and classifying sentiment from raw text.

    It was created with the intention to:
    - Educate and demonstrate the strengths and weaknesses of various ML models
    - Provide hands-on experience with real-time sentiment prediction
    - Offer a visually engaging way to compare model performances

    ---

    ##  Built With
    - **Programming Language**: Python 
    - **Web Framework**: [Streamlit](https://streamlit.io/)
    - **ML Libraries**: scikit-learn, pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn (optional)
    - **Preprocessing**: TF-IDF, SVD, BOW

    ---

    ##  Machine Learning Models Used
    -  **Decision Tree**
    -  **Logistic Regression**
    -  **K-Nearest Neighbors (KNN)**
    -  **Artificial Neural Network (ANN)**
    -  **Naive Bayes**
    -  **Support Vector Machine (SVM)**
    -  **Clustering (Unsupervised Learning)**

    Each model has been trained and tested on labeled sentiment datasets using traditional feature engineering techniques like **TF-IDF** and **SVD/PCA**.

    ---

    ##  Real-World Relevance
    Sentiment analysis is crucial in various fields such as:
    - **Marketing & Branding**: Analyze what customers think about a product
    - **Politics**: Gauge public opinion from tweets and news comments
    - **Finance**: Assess market sentiment from analyst blogs and news
    - **Customer Service**: Prioritize negative feedback and resolve faster
    - **Healthcare**: Analyze emotional state from patient messages

    ---

    ##  Author & Credits
    - **Developed by:** *Rudra Gupta(B23CS1098) , Aaditya Bansal(B23CS1083) , Om Sharma(B23CS1036) , Meet Tilala(B23CS1048) , Anmol Yadav(B23CS1004)*  
    - **Institution:** *Indian Institute of Technology Jodhpur*  
    - **GitHub:** https://github.com/OmChandraSharma/Sentiment_Analysis.git 
 

    Special thanks to open-source contributors and datasets used for training and testing.

    ---

    ##  Disclaimer
    This app is for educational and demonstration purposes only. It does not use any deep learning or transformer-based NLP techniques.

    ---

     _Feel free to explore the app and try live predictions using your own text!_
    """)
