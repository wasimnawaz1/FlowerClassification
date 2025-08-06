import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("🌸 Flower Classifier (5 Classes)")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(features)[0]
    st.success(f"🌼 Predicted Flower Class: {pred}")
