import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- CUSTOM CSS ----------
# st.markdown("""
#     <style>
#     /* Set background */
#     .stApp {
#         background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
#         background-size: cover;
#         color: white;
#         font-family: 'Segoe UI', sans-serif;
#     }

#     /* Sidebar styling */
#     .css-1d391kg, .css-1lcbmhc {
#         background-color: rgba(0, 0, 0, 0.7) !important;
#         color: white;
#     }

#     /* Title and headings */
#     h1, h2, h3 {
#         color: #ffffff;
#         text-shadow: 2px 2px 4px #000000;
#     }

#     /* Centering main content a bit */
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#         padding-left: 3rem;
#         padding-right: 3rem;
#     }

#     /* Buttons */
#     .stButton button {
#         background-color: #00bf72;
#         color: white;
#         border-radius: 12px;
#         padding: 0.5rem 1rem;
#         border: none;
#         transition: 0.3s ease-in-out;
#     }

#     .stButton button:hover {
#         background-color: #009e60;
#         transform: scale(1.05);
#     }
#     </style>
# """, unsafe_allow_html=True)


# ---------- LABELS ----------
label_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Sentiment App", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("Model Explorer")
page = st.sidebar.radio("Navigate to:", ["Home","Logistic Regression","KNN","Naive Bayes", "Decision Tree", "ANN", "Clustering", "Live Simulation","About"])

# ---------- HOME ----------
if page == "Home":
    import subpages.home as hm
    hm.render()

elif page == "About":
    import subpages.about as ab
    ab.render()

elif page == "Naive Bayes":
    import subpages.naive_bayes as nb
    nb.render()

elif page == "Linear Regression":
    import subpages.linear_regression as lb
    lb.render()

elif page == "Decision Tree":
    import subpages.decision_tree as dt
    dt.render()
    
elif page == "ANN":
    import subpages.ann as ann
    ann.render()
elif page == "KNN":
    import subpages.knn as knn
    knn.render()
elif page == "Clustering":
    import subpages.clustering as cl
    cl.render()

elif page == "Live Simulation":
    import subpages.simulation as sm
    sm.render()
    

# # ---------- PAGE CONFIG (must be first Streamlit command) ----------
# import streamlit as st
# st.set_page_config(page_title="Sentiment App", layout="wide")

# # ---------- STANDARD LIBRARY IMPORTS ----------
# import pandas as pd
# import numpy as np
# import re
# import joblib
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # ---------- LABEL MAPPING ----------
# label_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}


# # ---------- CUSTOM CSS ----------
# def load_custom_css():
#     st.markdown("""
#         <style>
#         /* Background - Light pastel gradient */
#         .stApp {
#             background: linear-gradient(to right, #fdfbfb, #ebedee);
#             color: #333333;
#             font-family: 'Segoe UI', sans-serif;
#         }

#         /* Sidebar - Soft light panel */
#         .css-1d391kg, .css-1lcbmhc, .css-6qob1r, .css-1cypcdb {
#             background-color: rgba(255, 255, 255, 0.8) !important;
#             color: #222222;
#             border-radius: 10px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.05);
#         }

#         /* Headings - Subtle shadows */
#         h1, h2, h3 {
#             color: #2c3e50;
#             text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
#         }

#         /* Main block container spacing */
#         .block-container {
#             padding-top: 2rem;
#             padding-bottom: 2rem;
#             padding-left: 3rem;
#             padding-right: 3rem;
#         }

#         /* Buttons - Soft green with hover */
#         .stButton button {
#             background-color: #91e5a9;
#             color: #222222;
#             border-radius: 8px;
#             padding: 0.5rem 1rem;
#             border: none;
#             transition: all 0.3s ease;
#         }

#         .stButton button:hover {
#             background-color: #6fd49b;
#             transform: scale(1.05);
#         }

#         /* Image styling if needed */
#         img {
#             border-radius: 10px;
#             box-shadow: 0 4px 10px rgba(0,0,0,0.05);
#         }
#         </style>
#     """, unsafe_allow_html=True)


# # ---------- MAIN APP ----------
# def main():
#     load_custom_css()

#     # Sidebar
#     st.sidebar.title("🧠 Model Explorer")
#     page = st.sidebar.radio("Navigate to:", [
#         "Home",
#         "Linear Regression",
#         "Naive Bayes",
#         "Decision Tree",
#         "ANN",
#         "Clustering",
#         "Simulation"
#     ])

#     # Pages
#     if page == "Home":
#         st.title("📊 Sentiment Analysis Models Overview")
#         st.write("""
#             Welcome to the Sentiment Analysis Dashboard. Here you can:
#             - View model performances
#             - Run a live text simulation with our Decision Tree model
#         """)
#         st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)

#     elif page == "Naive Bayes":
#         import subpages.naive_bayes as nb
#         nb.render()

#     elif page == "Linear Regression":
#         import subpages.linear_regression as lb
#         lb.render()

#     elif page == "Decision Tree":
#         import subpages.decision_tree as dt
#         dt.render()

#     elif page == "ANN":
#         import subpages.ann as ann
#         ann.render()

#     elif page == "Clustering":
#         import subpages.clustering as cl
#         cl.render()

#     elif page == "Simulation":
#         import subpages.simulation as sm
#         sm.render()


# if __name__ == "__main__":
#     main()
