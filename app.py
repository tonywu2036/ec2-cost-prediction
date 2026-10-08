import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Configuration
st.set_page_config(
    page_title="EC2 Cost Prediction",
    page_icon="☁️",
    layout="wide"
)

st.title("AWS EC2 Cost Prediction Dashboard")
st.write("Predict EC2 On-Demand costs using Linear Regression.")

# Load Dataset
data = pd.read_csv("ec2dataset.csv")

# Display Dataset
st.subheader("EC2 Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Total Instances", len(data))
col2.metric("Total Features", len(data.columns))

st.dataframe(data.head(10))

# Data Cleaning
data["On Demand"] = pd.to_numeric(
    data["On Demand"].str.replace(
        r"[$, hourly]", "", regex=True
    ),
    errors="coerce"
)

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"].str.extract(r"([\d.]+)")[0],
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

# Remove missing values
data_cleaned = data.dropna(
    subset=["On Demand", "Instance Memory", "vCPUs"]
)

# Prepare features and target
X = data_cleaned[["Instance Memory", "vCPUs"]]
y = data_cleaned["On Demand"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Display Model Performance
st.subheader("Linear Regression Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")

st.write(f"Training Samples: {len(X_train)}")
st.write(f"Testing Samples: {len(X_test)}")

# EC2 Cost Prediction
st.subheader("Predict EC2 Instance Cost")

col1, col2 = st.columns(2)

with col1:
    memory = st.number_input(
        "Instance Memory (GiB)",
        min_value=0.5,
        value=4.0,
        step=0.5
    )

with col2:
    vcpus = st.number_input(
        "Number of vCPUs",
        min_value=1,
        value=2,
        step=1
    )

# Prediction Button
if st.button("Predict Cost"):

    new_instance = pd.DataFrame(
        [[memory, vcpus]],
        columns=["Instance Memory", "vCPUs"]
    )

    predicted_cost = model.predict(new_instance)[0]

    st.write(f"Raw Model Prediction: ${predicted_cost:.4f}/hour")

    if predicted_cost < 0:
        st.warning(
            "The model predicted a negative cost. "
            "This is not a valid EC2 price and indicates "
            "a limitation of the Linear Regression model."
        )
    else:
        st.success(
            f"Predicted On-Demand Cost: ${predicted_cost:.4f}/hour"
        )

# Actual vs Predicted Visualization
st.subheader("Actual vs Predicted EC2 Costs")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(y_test, y_pred, alpha=0.6, color="blue")

ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

ax.set_title("Actual vs Predicted On-Demand Costs")
ax.set_xlabel("Actual Cost (USD/hour)")
ax.set_ylabel("Predicted Cost (USD/hour)")
ax.legend()

st.pyplot(fig)
plt.close(fig)

