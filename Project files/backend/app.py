from flask import Flask, request, jsonify, render_template
from ml_model import model, feature_columns
import numpy as np

app = Flask(__name__, template_folder="../frontend")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_data = request.json

        # Create full feature vector (26 features)
        input_dict = {feature: 0 for feature in feature_columns}

        # Map simple UI inputs to model features
        input_dict["founded_year"] = float(user_data.get("founded_year", 0))
        input_dict["funding_total_usd"] = float(user_data.get("funding_total_usd", 0))
        input_dict["milestones"] = float(user_data.get("milestones", 0))
        input_dict["relationships"] = float(user_data.get("relationships", 0))
        input_dict["has_VC"] = int(user_data.get("has_VC", 0))
        input_dict["has_angel"] = int(user_data.get("has_angel", 0))

        # Convert to ordered array (VERY IMPORTANT)
        input_array = np.array([input_dict[col] for col in feature_columns]).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        result = "Successful Startup 🚀" if prediction == 1 else "Failed Startup ❌"

        return jsonify({
            "prediction": result,
            "probability": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)