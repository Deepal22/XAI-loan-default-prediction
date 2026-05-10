import streamlit as st
import numpy as np
import joblib
import shap

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# SHAP Explainer
explainer = shap.TreeExplainer(model)


st.set_page_config(
    page_title="XAI Loan Default Prediction",
    layout="centered"
)


st.title("🏦 Explainable AI Loan Default Prediction")

st.write(
    "Predict loan default risk using XGBoost and explain predictions using SHAP."
)


st.sidebar.title("About Project")

st.sidebar.write(
    """
    This dashboard predicts loan default risk
    and explains which features influenced
    the prediction.
    """
)


st.header("Applicant Information")

income = st.number_input(
    "Applicant Income",
    min_value=1000.0,
    max_value=1000000.0,
    value=50000.0
)

credit = st.number_input(
    "Loan Amount",
    min_value=1000.0,
    max_value=1000000.0,
    value=200000.0
)

annuity = st.number_input(
    "Annuity Amount",
    min_value=1000.0,
    max_value=100000.0,
    value=15000.0
)

days_birth = st.number_input(
    "Days Birth (negative)",
    min_value=-30000,
    max_value=-7000,
    value=-12000
)

days_employed = st.number_input(
    "Days Employed (negative)",
    min_value=-20000,
    max_value=0,
    value=-3000
)

ext1 = st.slider(
    "EXT_SOURCE_1",
    0.0,
    1.0,
    0.5
)

ext2 = st.slider(
    "EXT_SOURCE_2",
    0.0,
    1.0,
    0.5
)

ext3 = st.slider(
    "EXT_SOURCE_3",
    0.0,
    1.0,
    0.5
)


if st.button("Predict Loan Risk"):

    # Create input array
    input_data = np.array([[
        income,
        credit,
        annuity,
        days_birth,
        days_employed,
        ext1,
        ext2,
        ext3
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]


    st.header("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠ High Risk of Default ({probability:.2f})"
        )

    else:

        st.success(
            f"✅ Low Risk of Default ({probability:.2f})"
        )


    st.header("Prediction Explanation")

    shap_values = explainer.shap_values(input_scaled)

    feature_names = [
        "Income",
        "Credit",
        "Annuity",
        "Birth Days",
        "Employment Days",
        "EXT1",
        "EXT2",
        "EXT3"
    ]

    shap_array = shap_values[0]

    # Sort by importance
    sorted_features = sorted(
        zip(feature_names, shap_array),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Generate explanation
    for feature, value in sorted_features:

        # Ignore very small impacts
        if abs(value) > 0.05:

            if value > 0:

                st.write(
                    f"🔺 {feature} increased the default risk."
                )

            else:

                st.write(
                    f"🔻 {feature} reduced the default risk."
                )


    st.info(
        "SHAP explanations show how each feature "
        "contributed to the prediction."
    )