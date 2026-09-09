import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load saved models
# --------------------------------------------------

regressor = joblib.load("used_car_regressor.pkl")
classifier = joblib.load("used_car_classifier.pkl")


# --------------------------------------------------
# Load dataset for dropdown options
# --------------------------------------------------

df = pd.read_csv(
    "Data/used_cars-selected-columns.csv"
)

# Same basic cleaning used during training
df["fuel_type"] = df["fuel_type"].replace(
    ["–", "not supported"],
    "Unknown"
)

df["fuel_type"] = df["fuel_type"].fillna("Unknown")
df["accident"] = df["accident"].fillna("Unknown")


# --------------------------------------------------
# Dropdown options
# --------------------------------------------------

brand_options = sorted(
    df["brand"].dropna().unique().tolist()
)

fuel_options = sorted(
    df["fuel_type"].dropna().unique().tolist()
)

transmission_options = sorted(
    df["transmission"].dropna().unique().tolist()
)

accident_options = sorted(
    df["accident"].dropna().unique().tolist()
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🚗 Used Car Price Predictor")

st.write(
    "Enter the vehicle details below to estimate its "
    "price and predicted price category."
)

st.divider()


# --------------------------------------------------
# User inputs
# --------------------------------------------------

brand = st.selectbox(
    "Brand",
    brand_options
)

mileage = st.number_input(
    "Mileage",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_options
)

transmission = st.selectbox(
    "Transmission",
    transmission_options
)

accident = st.selectbox(
    "Accident History",
    accident_options
)

horsepower = st.number_input(
    "Horsepower",
    min_value=0.0,
    value=200.0,
    step=10.0
)

engine_size = st.number_input(
    "Engine Size (Litres)",
    min_value=0.0,
    value=2.0,
    step=0.1
)

car_age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    value=5,
    step=1
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Car Price",
    use_container_width=True
):

    input_data = pd.DataFrame(
        {
            "brand": [brand],
            "mileage": [mileage],
            "fuel_type": [fuel_type],
            "transmission": [transmission],
            "accident": [accident],
            "horsepower": [horsepower],
            "engine_size": [engine_size],
            "car_age": [car_age]
        }
    )


    # Regression prediction
    predicted_price = regressor.predict(
        input_data
    )[0]


    # Classification prediction
    predicted_category = classifier.predict(
        input_data
    )[0]


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated Price",
            f"${predicted_price:,.2f}"
        )

    with col2:

        st.metric(
            "Price Category",
            predicted_category
        )


    st.success(
        "Prediction completed successfully."
    )

    st.caption(
        "This prediction is based on the machine learning "
        "models trained on the used-car dataset and should "
        "be treated as an estimate rather than an exact market price."
    )