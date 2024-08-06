import json
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Handle the data in csv file
def parse_csv_data(csv_string):
    lines = csv_string.strip().split('\n')
    header = lines[0].split(',')
    data = [line.split(',') for line in lines[1:]]
    return header, data

# Model creation based on the model name
def create_model(model_name):
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

def predict(csv_data, model_name, target_name, model_params):
    headers, rows = parse_csv_data(data["csv_data"])
    X, y, ids = prepare_data(rows, target_name, headers)
    odel = create_model(model_name, model_params)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Map predictions to ids
    results = [{"id": ids[i], "predicted_value": predictions[i]} for i in range(len(ids))]
    
    return {"predictions": results}

# Example input JSON
input_json = json.dumps({
    "csv_data": "id,name,age,income,target\n1,John,30,40000,1\n2,Jane,25,50000,0\n3,Bob,35,45000,1",
    "model_name": "logistic_regression",
    "target_name": "target"
    "model_params": {}
})

# API simulation
data = json.loads(input_json)
model_name = data["model_name"]
target_name = data["target_name"]
result = predict(data, model_name, target_name)

# Output the results
print(json.dumps(result, indent=2))

# Haven't done the http request handling part

