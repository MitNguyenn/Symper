from flask import Flask, request, jsonify
import pandas as pd
from NB import train as train_nb  # Import the train function from NB.py
from LinearReg import train as train_lr  # Import the train function from LinearReg.py

app = Flask(__name__)


# Function to create Naive Bayes model
def nb(data, target_name, model_params):
    model_name = data.get('model_name')
    sample_weight = model_params.get('sample_weight', None)
    alpha = model_params.get('alpha', 1.0)
    return train_nb(data, feature_names, target_name, model_name, sample_weight, alpha)

# Function to create Linear Regression model
def lr(data, feature_names, target_name):
    return train_lr(data, feature_names, target_name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieving the json file, getting elements from the json
        data = request.get_json()
        csv_data = data.get('csv_data')
        model_name = data.get('model_name').lower()
        target_name = data.get('target_name')
        feature_names = data.get('feature_names')
        model_params = data.get('model_params', {})

        # Return error if there are any fields missing
        if not csv_data or not model_name or not target_name or not feature_names:
            return jsonify({"error": "Missing required fields"}), 400

        # Converting from csv to data frame
        df = pd.read_csv(pd.compat.StringIO(csv_data))

        if 'naive bayes' in model_name:
            model = nb(df, feature_names, target_name, model_params)
            predictions = model.predict(df[feature_names])
            results = [{"id": df.index[i], "predicted_value": predictions[i]} for i in range(len(df))]

        elif model_name == 'linear regression':
            model, X_test, y_test = lr(df, feature_names, target_name)
            predictions = model.predict(X_test)
            results = [{"id": X_test.index[i], "predicted_value": predictions[i]} for i in range(len(X_test))]

        else:
            # Return error if the model name is invalid
            return jsonify({"error": "Unsupported model type"}), 400

        # Return results
        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Still missing function for logistics regression
