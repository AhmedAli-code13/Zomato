import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import gdown
import os
import importlib

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# تحقق من وجود مكتبة imbalanced-learn
try:
    import imblearn
except ModuleNotFoundError:
    st.warning()
    st.stop()

# إعداد واجهة التطبيق
st.set_page_config(page_title="Zomato Analysis App", layout="wide")

# ---------------------------
# تحميل البيانات والنموذج
# ---------------------------

@st.cache_resource
def load_data():
    csv_path = "old_zomato.csv"
    if not os.path.exists(csv_path):
        gdown.download("https://drive.google.com/uc?id=1YyK3bTvvSxHKnt2LOlk_55ffE18CKQI0",
                       csv_path, quiet=True)
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model():
    file_id = "1iCNobeEmtGfZmNLaywkf_WikFrXvbgYL"
    model_url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "RandomForest.pkl"

    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=True)

    return joblib.load(model_path)

# تحميل البيانات
old_df = load_data()
df = pd.read_csv("cleaned_df.csv", index_col=0)
rf_pipeline = load_model()

# ---------------------------
# واجهة التطبيق
# ---------------------------

page = st.sidebar.radio("Choose Page", ["Data & Analysis", "Prediction"])

# صفحة التحليل
if page == "Data & Analysis":
    st.title("Cleaned Data & Exploratory Analysis")

    st.subheader("Old Data")
    st.dataframe(old_df.head(20))
    st.markdown("---")

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(20))

    st.markdown("---")
    st.subheader("Exploratory Data Analysis")

    city_counts = df['listed_in(city)'].value_counts().nlargest(10).reset_index()
    city_counts.columns = ['city', 'count']
    fig1 = px.bar(city_counts, x='city', y='count', title="Top 10 Cities by Number of Restaurants")
    st.plotly_chart(fig1, use_container_width=True)

    cuisine_counts = df['cuisines'].value_counts().nlargest(10).reset_index()
    cuisine_counts.columns = ['cuisine', 'count']
    fig2 = px.bar(cuisine_counts, x='cuisine', y='count', title="Top 10 Popular Cuisines")
    st.plotly_chart(fig2, use_container_width=True)

    online_order_counts = df.groupby("online_order").size().reset_index(name="count")
    fig3 = px.pie(online_order_counts, names="online_order", values="count",
                 title="Online Order Availability")
    st.plotly_chart(fig3, use_container_width=True)

    avg_cost_city = df.groupby("listed_in(city)")["approx_cost(for two people)"].mean().nlargest(10).reset_index()
    fig4 = px.bar(avg_cost_city, x="listed_in(city)", y="approx_cost(for two people)",
                 title="Top 10 Cities by Average Cost for Two")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(df, x="votes", y="rate(?/5)", opacity=0.5,
                     title="Votes vs Rating (Scatter Plot)",
                     labels={"votes": "Number of Votes", "rate(?/5)": "Restaurant Rating"})
    st.plotly_chart(fig5, use_container_width=True)

# صفحة التنبؤ
else:
    st.title("Predict Restaurant Rating")

    col1, col2 = st.columns(2)
    with col1:
        online_order = st.selectbox("Online Order", df["online_order"].unique())
        book_table = st.selectbox("Book Table", df["book_table"].unique())
        location = st.selectbox("Location", df["location"].unique())
        rest_type = st.selectbox("Restaurant Type", df["rest_type"].unique())
    with col2:
        cuisines = st.selectbox("Cuisines", df["cuisines"].unique())
        listed_in_type = st.selectbox("Listed In (Type)", df["listed_in(type)"].unique())
        listed_in_city = st.selectbox("Listed In (City)", df["listed_in(city)"].unique())
        votes = st.number_input("Number of Votes", min_value=0, value=50)
        cost = st.number_input("Approx Cost for Two", min_value=50, value=500)

    if st.button("Predict Rating"):
        input_data = pd.DataFrame([{
            "online_order": online_order,
            "book_table": book_table,
            "votes": votes,
            "location": location,
            "rest_type": rest_type,
            "cuisines": cuisines,
            "approx_cost(for two people)": cost,
            "listed_in(type)": listed_in_type,
            "listed_in(city)": listed_in_city
        }])
        pred = rf_pipeline.predict(input_data)[0]
        st.success(f"Predicted Restaurant Rating: {round(pred,1)} / 5")
