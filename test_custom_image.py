import numpy as np
from PIL import Image, ImageOps
import joblib
import os
import matplotlib.pyplot as plt

# Load trained neural network model
model_path = "model/digit_model_nn.pkl"
if not os.path.exists(model_path):
    print("Trained neural network model not found. Run app.py first to train it.")
    exit()

model = joblib.load(model_path)

# Load and process custom image
image_path = "custom_digit.png"
if not os.path.exists(image_path):
    print("custom_digit.png not found in the project folder.")
    exit()

# Convert to grayscale and invert (black on white)
img = Image.open(image_path).convert("L")
img = ImageOps.invert(img)
img = img.resize((8, 8))  # Resize to 8x8 pixels

# Scale pixel values to 0–16 +
img_array = np.array(img)
img_scaled = (img_array / 255.0) * 16.0
img_scaled = img_scaled.astype(np.float64)

# Flatten and predict
input_data = img_scaled.flatten().reshape(1, -1)
prediction = model.predict(input_data)[0]

# Show result
print(f"✅ Predicted Digit: {prediction}")
plt.imshow(img_scaled, cmap="gray")
plt.title(f"Predicted: {prediction}")
plt.axis("off")
plt.show()
