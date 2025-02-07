from flask import Flask, request, jsonify, render_template
import pandas as pd
from src.predict import ModelPredictor
import joblib 

app = Flask(__name__)

# Load the saved model
model_path = 'rfc_model.pkl'
model = joblib.load(model_path)
predictor = ModelPredictor(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Predict sentiments
        result_df = predictor.predict_df(df, text_column='text')
        
        # Convert the result DataFrame to HTML table
        result_html = result_df.to_html(index=False)
        
        return render_template('result.html', result=result_html)

if __name__ == '__main__':
    app.run(debug=True)