
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression',
    'LassoRegression', 'ElasticNet', 'PolynomialRegression',
    'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM',
    'XGBoost', 'KNN'
]

# Use the directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

models = {}

for name in model_names:
    model_path = BASE_DIR / f'{name}.pkl'

    if not model_path.is_file():
        raise FileNotFoundError(f'Model file not found: {model_path}')

    with model_path.open('rb') as file:
        models[name] = pickle.load(file)

# Load evaluation results
results_path = BASE_DIR / 'model_evaluation_results.csv'
results_df = pd.read_csv(results_path)


@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)


@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model')

    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 400

    feature_names = [
        'Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population'
    ]

    try:
        input_data = {
            feature: float(request.form[feature])
            for feature in feature_names
        }
    except (KeyError, ValueError):
        return jsonify({
            'error': 'Please provide valid numeric values for all features.'
        }), 400

    input_df = pd.DataFrame([input_data], columns=feature_names)

    try:
        model = models[model_name]
        prediction = model.predict(input_df)[0]

        return render_template(
            'results.html',
            prediction=prediction,
            model_name=model_name
        )
    except (ValueError, TypeError) as error:
        app.logger.exception('Prediction failed')
        return jsonify({
            'error': f'Prediction failed. Check model features and preprocessing: {error}'
        }), 400


@app.route('/results')
def results():
    return render_template(
        'model.html',
        tables=[results_df.to_html(classes='data', index=False)],
        titles=results_df.columns.values
    )


if __name__ == '__main__':
    app.run(debug=True)