import json
import numpy as np
from main import app
import random
import joblib

def test():
    app.config['TESTING'] = True
    client = app.test_client()

    def generate_random_data(rows, cols):
        data = np.random.uniform(1.0, 10.0, size=(rows, cols)).tolist()
        columns = [f"col{i+1}" for i in range(cols)]
        return [columns] + data
    
    def get_model_feature_names(model_id):
        model = joblib.load(f"save/{model_id}.pkl")
        return model.feature_names_in_


    def test_train_linear_regression():
        try:
            data = {
                "data": generate_random_data(100, 5),
                "target": ["col1"],
                "parameters": {
                    "test_size": 0.2,
                    "fit_intercept": True,
                    "positive": True
                }
            }
            response = client.post('/train/linear_regression', data=json.dumps(data), content_type='application/json')
            response_data = response.json
            print("Linear Regression Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'model_id' in response_data
            assert 'evaluation' in response_data
        except AssertionError as e:
            print(f"Linear Regression Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Linear Regression Test: {e}")

    def test_train_naive_bayes():
        try:
            data = generate_random_data(100, 4)
            target = np.random.choice([1, 2], size=100).tolist()
            for i in range(1, len(data)):
                data[i][0] = target[i-1]

            json_file = {
                "data": data,
                "target": ["col1"],
                "parameters": {
                    "test_size": 0.2,
                    "model_type": random.choice(["gaussian", "multinomial", "bernoulli"]),
                    "alpha": 1e-9
                }
            }
            response = client.post('/train/naive_bayes', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            print("Naive Bayes Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'model_id' in response_data
            assert 'evaluation' in response_data
        except AssertionError as e:
            print(f"Naive Bayes Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Naive Bayes Test: {e}")


    def test_train_logistics_regression():
        try:
            data = generate_random_data(100, 5)
            target = np.random.choice([1, 2], size=100).tolist()
            for i in range(1, len(data)):
                data[i][0] = target[i-1]

            json_file = {
                "data": data,
                "target": ["col1"],
                "parameters": {
                    "test_size": 0.2,
                    "penalty": "l2",
                    "tol": 1e-4,
                    "C": 1.0,
                    "fit_intercept": True
                }
            }
            response = client.post('/train/logistics_regression', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            print("Logistic Regression Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'model_id' in response_data
            assert 'evaluation' in response_data
        except AssertionError as e:
            print(f"Logistic Regression Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Logistic Regression Test: {e}")

    def test_predict():
        try:
            model_id = "['col1']`~505cfc79-d119-435a-b595-6188cc905930"
            model_features = get_model_feature_names(model_id)
            data = generate_random_data(100, len(model_features))
            data[0] = model_features.tolist()
            json_file = {
                "data": data,
                "model_id": model_id
            }
            response = client.post('/predict', data=json.dumps(json_file), content_type='application/json')
            response_data = response.json
            print("Prediction Test Response:", response_data)

            assert response.status_code == 200
            assert response_data['status'] == 'OK'
            assert 'prediction' in response_data
        except AssertionError as e:
            print(f"Prediction Test Failed: {e}")
        except Exception as e:
            print(f"An error occurred in Prediction Test: {e}")

    test_train_linear_regression()
    test_train_naive_bayes()
    test_train_logistics_regression()
    test_predict()

if __name__ == '__main__':
    test()
