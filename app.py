import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from model import predict
import datetime

st.title("Global Store Sales Prediction")
st.markdown("Machine Learning untuk memprediksi total Purchase berdasarkan tanggal")

st.header("Input Tanggal")

col1, col2 = st.columns(2)

with col1:
    day = st.slider("Tanggal (Hari)", 1, 31, 15)
    month = st.slider("Bulan", 1, 12, 6)

with col2:
    year = st.slider("Tahun", min_value=2016, max_value=2017)
    object = datetime.date(year, month, day)
    weekDay = object.weekday()
    st.text("Day of Week: " + str(weekDay))

st.text("")

if st.button("Predict Purchase") :
    input = np.array([[day, month, year, weekDay]])
    result = predict(input)
    st.success(f"Total Pembelian: ${result[0]:.2f}")

st.text("")
st.markdown("---")
st.markdown("Made by Arya Pannadana - 2025")
