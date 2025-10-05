import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the route statistics and model
route_stats = joblib.load("state_route_stats.joblib")
model = joblib.load('co2_model.joblib')

# Load dataset for dropdowns
df = pd.read_csv('delhivery.csv')

# Extract cities from location strings
def extract_city(location):
    if isinstance(location, str):
        return location.split(",")[0].strip()
    return ""

df["pickup_city"] = df["source_name"].apply(extract_city)
df["drop_city"] = df["destination_name"].apply(extract_city)

# Get unique city options
pickup_cities = sorted(df["pickup_city"].unique())
drop_cities = sorted(df["drop_city"].unique())

# --- Streamlit App Layout ---
st.set_page_config(page_title="CO₂ Emissions Predictor", layout="centered")

st.title("🚚 CO₂ Emissions & Cost Estimator")
st.markdown("Predict logistics emissions and estimated cost between Indian cities.")

st.subheader("📍 Route Selection")
pickup_city = st.selectbox("Select Pickup City", pickup_cities)
drop_city = st.selectbox("Select Drop City", drop_cities)

st.subheader("📦 Package Details")
weight = st.number_input("Package Weight (kg)", min_value=0.1, max_value=5000.0, value=10.0, step=0.1)

# --- Prediction ---
if st.button("🔍 Predict CO₂ Emissions"):
    route_key = f"{pickup_city}_{drop_city}"

    if route_key in route_stats:
        route_info = route_stats[route_key]
        avg_distance = route_info.get("distance", 300.0)
        avg_time = route_info.get("time", 5.0)
    else:
        st.warning("⚠️ No historical data for this route. Using average fallback values.")
        all_distances = [v["distance"] for v in route_stats.values() if "distance" in v]
        all_times = [v["time"] for v in route_stats.values() if "time" in v]
        avg_distance = np.mean(all_distances) if all_distances else 300.0
        avg_time = np.mean(all_times) if all_times else 5.0

    # Prepare model input
    input_data = pd.DataFrame([[avg_distance, avg_time, weight]],
                              columns=["estimated_distance", "estimated_time", "Weight"])

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Results")
    if isinstance(prediction, (list, np.ndarray)) and len(prediction) == 2:
        co2, cost = prediction
        st.success(f"🌍 **Predicted CO₂ Emissions:** `{co2:.2f} kg`")
        st.info(f"💰 **Estimated Cost:** `₹{cost:.2f}`")
    else:
        st.success(f"🌍 **Predicted CO₂ Emissions:** `{prediction:.2f} kg`")
