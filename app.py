from flask import Flask, render_template, request
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import joblib

app = Flask(__name__)

# Load dataset
digits = load_digits()

# Ensure folders exist
os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Model path
model_path = "model/digit_model_nn.pkl"

# Load or train model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    true_value = None
    index = None

    if request.method == "POST":
        index = int(request.form["index"])
        image = digits.images[index]
        data = digits.data[index].reshape(1, -1)

        prediction = model.predict(data)[0]
        true_value = digits.target[index]

        # Save the digit image with prediction
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {true_value}, Predicted: {prediction}")
        ax.axis('off')
        fig.savefig("static/result.png")
        plt.close(fig)

    return render_template("index.html", prediction=prediction, true_value=true_value, index=index)

if __name__ == "__main__":
    app.run(debug=True)
