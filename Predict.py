import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Load MNIST test data
(_, _), (X_test, y_test) = mnist.load_data()

# Select one test image
index = 10

img = X_test[index]

# Show image
plt.imshow(img, cmap="gray")
plt.title("Actual Digit: " + str(y_test[index]))
plt.axis("off")
plt.show()

# Preprocess image
img = img / 255.0

img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)

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

digit = np.argmax(prediction)

print("Predicted Digit:", digit)
print("Actual Digit:", y_test[index])   

word = digit_words[digit]

print("Predicted Digit:", digit)
print("Predicted Word:", word)
print("Actual Digit:", y_test[index])