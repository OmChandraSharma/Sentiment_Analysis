import streamlit as st
from utils.preprocess import clean_text
from utils.load_models import load_decision_tree_models
from utils.load_models import load_naive_bayes
from utils.load_models import load_clusterring
from utils.load_models import load_KNN
from utils.load_models import load_logistic_regression
from utils.load_models import load_ann
from utils.load_models import load_svm

# def render():
#     dt_model, dt_vectorizer, dt_svd, dt_label_encoder = load_decision_tree_models()
#     nb_model, nb_vectorizer, nb_svd, nb_label_encoder = load_naive_bayes()
#     print("Naive Bayes Model:", nb_model)


#     st.title("🚀 Live Sentiment Simulation")

#     # Decision Tree Section
#     st.markdown("## Using Decision Tree")
#     user_input_dt = st.text_area("📝 Your Text for Decision tree:")
#     if st.button("Predict Sentiment1"):
#         if dt_model and dt_vectorizer and dt_label_encoder:
#             if user_input_dt.strip() == "":
#                 st.warning("Please enter some text.")
#             else:
#                 cleaned = clean_text(user_input_dt)
#                 st.write("✅ **Cleaned Text:**", cleaned)
#                 try:
#                     X_vec = dt_vectorizer.transform([cleaned])
#                     X_reduced = dt_svd.transform(X_vec)
#                     pred = dt_model.predict(X_reduced)[0]
#                     pred_label = dt_label_encoder.inverse_transform([pred])[0]
#                     st.success(f"🎯 Predicted Sentiment: **{pred_label}**")
#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")
#         else:
#             st.error("Models not loaded. Check paths.")

#     # Naive Bayes Section
#     st.markdown("## Using Naive Bayes")
#     user_input_nb = st.text_area("📝 Your Text for Naive Bayes:")
#     if st.button("Predict Sentiment2"):
#         if nb_model and nb_vectorizer and nb_label_encoder:
#             if user_input_nb.strip() == "":
#                 st.warning("Please enter some text.")
#             else:
#                 cleaned = clean_text(user_input_nb)
#                 st.write("✅ **Cleaned Text:**", cleaned)
#                 try:
#                     X_vec = nb_vectorizer.transform([cleaned])
#                     pred = nb_model.predict(X_vec)[0]
#                     # pred_label = nb_label_encoder.inverse_transform([pred])[0]
#                     st.success(f"🎯 Predicted Sentiment: **{pred}**")
#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")
#         else:
#             st.error("Models not loaded. Check paths.")

def render():
    # Load all models
    dt_model, dt_vectorizer, dt_svd, dt_label_encoder = load_decision_tree_models()
    nb_model, nb_vectorizer, nb_svd, nb_label_encoder = load_naive_bayes()
    ann_model, ann_vectorizer, ann_svd, ann_label_encoder = load_ann()

    st.title("🚀 Live Sentiment Classification Simulation")
    st.markdown("### Enter your text and see predictions from all models:")

    st.title("🚀 Live Sentiment Classification Simulation")
    st.markdown("### Enter your text and see predictions from both models:")
    user_input = st.text_area("📝 Your Text:")
    
    if st.button("🎯 Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            st.write("✅ **Cleaned Text:**", cleaned)

            # --- Decision Tree Prediction ---
            st.markdown("#### Decision Tree Prediction")
            if dt_model and dt_vectorizer and dt_label_encoder:
                try:
                    X_vec = dt_vectorizer.transform([cleaned])
                    X_reduced = dt_svd.transform(X_vec)
                    pred_dt = dt_model.predict(X_reduced)[0]
                    pred_label_dt = dt_label_encoder.inverse_transform([pred_dt])[0]
                    st.success(f"Decision Tree Sentiment: **{pred_label_dt}**")
                except Exception as e:
                    st.error(f"❌ Decision Tree prediction failed: {e}")
                else:
                    st.error("❗ Decision Tree model not loaded properly.")

                # --- Naive Bayes Prediction ---
                st.markdown("#### Naive Bayes Prediction")
                if nb_model and nb_vectorizer and nb_label_encoder:
                    try:
                        X_vec_nb = nb_vectorizer.transform([cleaned])
                        pred_nb = nb_model.predict(X_vec_nb)[0]
                        # pred_label_nb = nb_label_encoder.inverse_transform([pred_nb])[0]
                        st.success(f"Naive Bayes Sentiment: **{pred_nb}**")
                    except Exception as e:
                        st.error(f"❌ Naive Bayes prediction failed: {e}")
                else:
                    st.error("❗ Naive Bayes model not loaded properly.")


            # # --- ANN Prediction ---
            # st.markdown("#### ANN Prediction")
            # if ann_model and ann_vectorizer and ann_label_encoder:
            #     try:
            #         X_vec_ann = ann_vectorizer.transform([cleaned])
            #         X_reduced_ann = ann_svd.transform(X_vec_ann)
            #         pred_ann = ann_model.predict(X_reduced_ann)
            #         pred_label_ann = ann_label_encoder.inverse_transform([pred_ann.argmax(axis=1)][0])[0]
            #         st.success(f"ANN Sentiment: **{pred_label_ann}**")
            #     except Exception as e:
            #         st.error(f"❌ ANN prediction failed: {e}")
            # else:
            #     st.error("❗ ANN model not loaded properly.")
