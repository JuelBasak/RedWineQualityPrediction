# Importing the required libraries
from flask import Flask, render_template, request
import joblib

#Creating the Flask app
app = Flask(__name__)

# Returing the homepage when this API's Endpoint is hit
@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

"""
This is the predict method which predicts the quality of the Red Wine. 
The data is read from the homepage's form and is stored in this API. 
The prediction model is loaded (Decision Tree in this case) and the stored
values are passed into the model and the model returns the predicted value.
This predicted value is passed to the output Webpage and the result is shown using Jinja. 
"""

@app.route("/predict", methods=['POST'])
def predict():
    fixed_acidity = request.form['fixed_acidity']
    volatile_acidity = request.form['volatile_acidity']
    citric_acid = request.form['citric_acid']
    residual_sugar = request.form['residual_sugar']
    chlorides = request.form['chlorides']
    free_sulfur_dioxide = request.form['free_sulfur_dioxide']
    total_sulfur_dioxide = request.form['total_sulfur_dioxide']
    density = request.form['density']
    pH = request.form['pH']
    sulphates = request.form['sulphates']
    alcohol = request.form['alcohol']

    model = joblib.load('./model/new_tree.pkl')

    prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    if prediction[0] < 5:
        return render_template('output1.html', prediction=prediction[0])
    return render_template('output2.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=False)
