from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    print('Working on homepage method')
    return render_template('index.html')

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
