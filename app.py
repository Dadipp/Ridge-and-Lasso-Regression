import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("e:/VScode/HANDSON_32B/my_portfolio/Regression/data/boston.csv")
    return df

st.title("🏘️ Ridge & Lasso Regression on Boston Housing Dataset")

# Load and show data
df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())

# Split features and target
X = df.drop("medv", axis=1)
y = df["medv"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar for model selection
st.sidebar.title("Model Settings")
model_type = st.sidebar.selectbox("Choose regression type", ["Ridge", "Lasso"])
alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.01)

# Model training
if model_type == "Ridge":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
st.subheader("Model Evaluation")
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R2 Score:", r2_score(y_test, y_pred))

# Feature input prediction
st.subheader("Predict Housing Price")
input_data = []
for feature in X.columns:
    val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)
prediction = model.predict(input_scaled)

st.write("### Predicted MEDV (Price):", prediction[0])