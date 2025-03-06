from flask import Flask, render_template, request, jsonify, redirect
import numpy as np
import pickle
import time

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))

# Define the home route
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Collect form data from the user
    data = {
        "age": int(request.form.get("age")),
        "heart_rate": int(request.form.get("heart_rate")),
        "is_diabetic": int(request.form.get("is_diabetic")),
        "family_heart_problem_background": int(
            request.form.get("family_heart_problem_background")
        ),
        "is_smoker": int(request.form.get("smoker")),
        "is_alcohol": int(request.form.get("is_alcohol")),
        "exercise_time": int(request.form.get("exercise")),
        "diet": int(request.form.get("diet")),
    }
    print(data)  # Debugging: Print data to console

    # Convert data into a NumPy array for model prediction
    data_array = np.array(
        [
            [
                data["age"],
                data["heart_rate"],
                data["is_diabetic"],
                data["family_heart_problem_background"],
                data["is_smoker"],
                data["is_alcohol"],
                data["exercise_time"],
                data["diet"],
            ]
        ]
    )

    # Make prediction using the loaded model
    pred = model.predict(data_array)
    time.sleep(5)  # Simulate processing delay for user experience

    # Redirect based on prediction result
    if pred == 0:
        return redirect("/success")
    return redirect("/failure")

# Define the success page route
@app.route("/success", methods=["GET"])
def success():
    return render_template("success.html")

# Define the failure page route
@app.route("/failure", methods=["GET"])
def failure():
    return render_template("failure.html")

# Define a custom 404 error page route
@app.route("/404", methods=["GET"])
def error():
    return render_template("404.html")

# Catch-all route for undefined paths, redirecting to 404 page
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    return redirect("/404")

# Run the app (if running directly)
if __name__ == "__main__":
    app.run(debug=True)