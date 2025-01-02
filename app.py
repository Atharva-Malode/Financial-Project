from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "random_forest_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict using the model
        predictions = model.predict(input_df)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
