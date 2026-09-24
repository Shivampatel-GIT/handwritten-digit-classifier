import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Load trained CNN model
# -----------------------------
model = tf.keras.models.load_model("digit_model.keras")

# Digit to word
digit_words = {
    0: "ZERO",
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
    6: "SIX",
    7: "SEVEN",
    8: "EIGHT",
    9: "NINE"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("✍️ Handwritten Digit Recognition")

st.write("Draw a digit from 0 to 9 in the box below.")

canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Digit"):

    if canvas.image_data is not None:

        # Get canvas image
        image = canvas.image_data

        # Convert RGBA to grayscale
        image = image[:, :, :3]
        image = np.mean(image, axis=2)

        # Resize to MNIST size
        image = tf.image.resize(
            image[..., np.newaxis],
            [28, 28]
        ).numpy()

        # Normalize
        image = image / 255.0

        # Add batch dimension
        image = image.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(image, verbose=0)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        word = digit_words[digit]

        # Display result
        st.success(f"Predicted Digit: {digit}")
        st.success(f"Predicted Word: {word}")

        st.info(f"Confidence: {confidence:.2f}%")