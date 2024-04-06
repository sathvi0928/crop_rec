from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Split dataset into features and labels
X = data.drop(columns=['label'])
y = data['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        features = request.form.to_dict()
        # Convert data to list for prediction
        new_data = [[float(features['N']), float(features['P']), float(features['K']), float(features['temperature']), float(features['humidity']), float(features['ph']), float(features['rainfall'])]]
        # Predict crop
        predicted_crop = model.predict(new_data)
        return render_template('result.html', predicted_crop=predicted_crop[0])


if __name__ == '__main__':
    app.run(debug=True)
