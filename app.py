import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Crop Health AI",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# 38 CLASS NAMES
# ==========================================

classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "plant_disease_model.keras"
    )


model = load_model()


# ==========================================
# WEATHER RISK FUNCTION
# ==========================================

def weather_risk(
    temperature,
    humidity,
    rainfall,
    disease
):

    risk_score = 0

    if humidity >= 80:
        risk_score += 2

    elif humidity >= 60:
        risk_score += 1

    if rainfall >= 20:
        risk_score += 2

    elif rainfall >= 5:
        risk_score += 1

    if 20 <= temperature <= 30:
        risk_score += 1

    disease_lower = disease.lower()

    if any(word in disease_lower for word in [
        "blight",
        "mildew",
        "rust",
        "bacterial"
    ]):

        risk_score += 1

    if risk_score >= 5:
        risk = "HIGH"

    elif risk_score >= 3:
        risk = "MODERATE"

    else:
        risk = "LOW"

    return risk_score, risk


# ==========================================
# ADVISORY FUNCTION
# ==========================================

def generate_advisory(
    disease,
    risk
):

    disease_lower = disease.lower()

    if "late_blight" in disease_lower:

        if risk == "HIGH":

            return (
                "High crop-health risk detected for Tomato Late Blight. "
                "Inspect nearby plants immediately, remove severely affected "
                "leaves, avoid prolonged leaf wetness, and consult an "
                "agricultural expert for suitable disease-management treatment."
            )

        return (
            "Monitor tomato plants regularly for dark lesions and leaf damage. "
            "Maintain good field ventilation and avoid unnecessary leaf wetness."
        )

    elif "early_blight" in disease_lower:

        return (
            "Monitor the crop for increasing leaf spots. Remove severely "
            "affected plant material and maintain good field sanitation."
        )

    elif "rust" in disease_lower:

        return (
            "Inspect surrounding plants for rust symptoms and monitor disease "
            "spread. Maintain field sanitation and seek expert guidance if "
            "symptoms increase."
        )

    elif "bacterial" in disease_lower:

        return (
            "Monitor for bacterial symptoms and avoid unnecessary handling of "
            "affected plants. Remove severely affected material and consult "
            "an agricultural expert."
        )

    elif "healthy" in disease_lower:

        return (
            "The crop appears healthy. Continue regular monitoring and maintain "
            "proper irrigation, nutrition and field sanitation."
        )

    else:

        return (
            "Continue monitoring the crop regularly. For uncertain or severe "
            "symptoms, consult an agricultural expert or laboratory."
        )


# ==========================================
# EXPERT REFERRAL
# ==========================================

def expert_referral(
    confidence
):

    if confidence < 0.60:

        return (
            "Prediction confidence is low. "
            "Refer the sample to an agricultural expert or laboratory "
            "for confirmation."
        )

    elif confidence < 0.80:

        return (
            "Prediction confidence is moderate. "
            "Continue field monitoring and consider expert verification."
        )

    else:

        return (
            "Prediction confidence is high. "
            "The result can be used for preliminary crop-health assessment. "
            "Continue monitoring."
        )


# ==========================================
# HEADER
# ==========================================

st.title("🌱 Crop Health AI")

st.subheader(
    "AI-Based Crop Disease Detection and Risk Assessment"
)

st.write(
    "Upload a crop leaf image and provide environmental conditions "
    "to obtain a preliminary crop-health assessment."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("🌦️ Weather Information")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

rainfall = st.sidebar.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=5.0
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📷 Upload a crop leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        width=400
    )

    # Resize to model input size

    image_resized = image.resize(
        (128, 128)
    )

    # Convert to NumPy

    image_array = np.array(
        image_resized
    )

    # Normalize

    image_array = image_array / 255.0

    # Add batch dimension

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ======================================
    # CNN PREDICTION
    # ======================================

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    confidence = float(
        np.max(prediction[0])
    )

    predicted_disease = classes[
        predicted_index
    ]


    # ======================================
    # DISPLAY PREDICTION
    # ======================================

    st.divider()

    st.header("🔍 AI Disease Detection")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Disease",
            predicted_disease.replace(
                "___",
                " - "
            )
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ======================================
    # WEATHER RISK
    # ======================================

    risk_score, risk = weather_risk(
        temperature,
        humidity,
        rainfall,
        predicted_disease
    )

    st.header("🌦️ Weather Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Risk Score",
            risk_score
        )

    with col2:

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Humidity",
            f"{humidity}%"
        )


    # ======================================
    # CONFIDENCE CHECK
    # ======================================

    if confidence < 0.60:

        st.warning(
            "⚠️ LOW CONFIDENCE"
        )

        st.write(
            expert_referral(
                confidence
            )
        )

        st.info(
            "Disease-specific advisory is withheld because "
            "the model confidence is low."
        )

    else:

        # ==================================
        # ADVISORY
        # ==================================

        st.header("📋 Crop Health Advisory")

        advisory = generate_advisory(
            predicted_disease,
            risk
        )

        st.info(
            advisory
        )


        # ==================================
        # EXPERT VALIDATION
        # ==================================

        st.header("👨‍🌾 Expert Validation")

        st.success(
            expert_referral(
                confidence
            )
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Student prototype for preliminary crop-health assessment. "
    "Expert or laboratory confirmation should be obtained before "
    "disease-specific treatment decisions."
)