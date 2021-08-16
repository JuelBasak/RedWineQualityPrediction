from Flask import flask

app = flask(__name__)

@app.route('/')
def homepage():
    pass
def predict():
    pass

if __name__ == '__main__':
    app.run(debug=True)
