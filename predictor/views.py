import os
import joblib
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models/naive_bayes_model.pkl")

# Load the saved model
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Prediction route
def predict(request):
    model = load_model()
    if not model:
        return JsonResponse({"error": "Model not found. Train it first by running train.py."}, status=400)

    # Initialize context
    context = {
        "prediction": None,  # Default to None
        "form_data": {}      # Empty dict for form persistence
    }

    if request.method == "POST":
        # Extract and store form data
        form_data = {
            "Pregnancies": request.POST.get("Pregnancies", ""),
            "Glucose": request.POST.get("Glucose", ""),
            "BloodPressure": request.POST.get("BloodPressure", ""),
            "SkinThickness": request.POST.get("SkinThickness", ""),
            "Insulin": request.POST.get("Insulin", ""),
            "BMI": request.POST.get("BMI", ""),
            "DiabetesPedigreeFunction": request.POST.get("DiabetesPedigreeFunction", ""),
            "Age": request.POST.get("Age", "")
        }
        context["form_data"] = form_data

        try:
            # Convert to float for prediction
            input_array = np.array([
                float(form_data["Pregnancies"]),
                float(form_data["Glucose"]),
                float(form_data["BloodPressure"]),
                float(form_data["SkinThickness"]),
                float(form_data["Insulin"]),
                float(form_data["BMI"]),
                float(form_data["DiabetesPedigreeFunction"]),
                float(form_data["Age"])
            ]).reshape(1, -1)

            # Make the prediction
            prediction = model.predict(input_array)[0]
            # Map prediction to human-readable output
            prediction_text = "Diabetic" if prediction == "tested_positive" else "Non-Diabetic"
            
            # Store prediction and form data in session
            request.session["prediction"] = prediction_text
            request.session["form_data"] = form_data
            
            # Redirect to clear POST state
            return redirect("predict")
        except ValueError as e:
            context["prediction"] = f"Error: Invalid input ({str(e)})"
            return render(request, "index.html", context)

    # Handle GET request (including after redirect)
    if "prediction" in request.session:
        context["prediction"] = request.session.pop("prediction")  # Retrieve and remove
        context["form_data"] = request.session.pop("form_data", {})  # Retrieve and remove
    else:
        context["prediction"] = None
        context["form_data"] = {}

    return render(request, "index.html", context)