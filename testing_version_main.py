import json
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify


app = Flask(__name__)


# Handle the data in csv file
def parse_csv_data(csv_string):
    lines = csv_string.strip().split('\n')
    header = lines[0].split(',')
    data = [line.split(',') for line in lines[1:]]
    return header, data

# Model creation based on the model name
def create_model(model_name, model_params):
    if model_name == "linear_regression":
        return LinearRegression(**model_params)
    elif model_name == "logistic_regression":
        return LogisticRegression(**model_params)
    elif model_name == "naive_bayes":
        return GaussianNB(**model_params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported")

# Split data in to X, y and id to then later maps the output to the id
def prepare_data(data, target_name, headers):
    target_index = headers.index(target_name)
    id_index = headers.index("id")
    X = []
    y = []
    ids = []
    for row in data:
        ids.append(row[id_index])
        y.append(float(row[target_index]))
        X.append([float(row[i]) for i in range(len(row)) if i != target_index and i != id_index])
    return np.array(X), np.array(y), ids

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the necessary information
        csv_data = data.get('csv_data')
        model_name = data.get('model_name')
        target_name = data.get('target_name')
        model_params = data.get('model_params', {})

        if not csv_data or not model_name or not target_name:
            return jsonify({"error": "Missing required fields"}), 400

        # Parse the CSV data and prepare the dataset
        headers, rows = parse_csv_data(csv_data)
        X, y, ids = prepare_data(rows, target_name, headers)

        # Create and train the model
        model = create_model(model_name, model_params)
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)
        results = [{"id": ids[i], "predicted_value": predictions[i]} for i in range(len(ids))]

        # Return the results as a JSON response
        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define and start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
