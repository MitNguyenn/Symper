import unittest
import json
from flask import Flask
from flask_testing import TestCase
from main import app  # Import the Flask app

class TestAPI(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_train_linear_regression(self):
        data = {
            "data": [
                ["col1", "col2", "col3", "col4", "col5"],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0]
            ],
            "target": ["col1"],
            "parameters": {
                "test_size": 0.2,
                "fit_intercept": True,
                "positive": True
            }
        }
        response = self.client.post('/train/linear_regression', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.json
        print(response_data)
        self.assertEqual(response_data['status'], 'OK')
        self.assertIn('model_id', response_data)
        self.assertIn('evaluation', response_data)

    def test_train_naive_bayes(self):
        data = {
            "data": [
                ["col1", "col2", "col3", "col4", "col5"],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0]
            ],
            "target": ["col1"],
            "parameters": {
                "test_size": 0.2,
                "type": "gaussian",
                "alpha": 1e-9
            }
        }
        response = self.client.post('/train/naive_bayes', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.json
        print(response_data)
        self.assertEqual(response_data['status'], 'OK')
        self.assertIn('model_id', response_data)
        self.assertIn('evaluation', response_data)

    def test_train_logistics_regression(self):
        data = {
            "data": [
                ["col1", "col2", "col3", "col4", "col5"],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 4.0, 5.0, 6.0, 7.0],
                [2.0, 5.0, 6.0, 7.0, 8.0]
            ],
            "target": ["col1"],
            "parameters": {
                "test_size": 0.2,
                "penalty": "l2",
                "tol": 1e-4,
                "C": 1.0,
                "fit_intercept": True
            }
        }
        response = self.client.post('/train/logistics_regression', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.json
        print(response_data)
        self.assertEqual(response_data['status'], 'OK')
        self.assertIn('model_id', response_data)
        self.assertIn('evaluation', response_data)

    def test_predict(self):
        data = {
            "data": [
                ["col1", "col2", "col3", "col4", "col5"],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0]
            ],
            "model_id": "col1~3de6afeb-8e57-4b68-8a05-0d224918684f"
        }
        response = self.client.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.json
        print(response_data)
        self.assertEqual(response_data['status'], 'OK')
        self.assertIn('prediction', response_data)

if __name__ == '__main__':
    unittest.main()
