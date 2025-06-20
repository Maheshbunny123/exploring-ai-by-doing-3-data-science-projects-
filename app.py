from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained ML model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = news.lower()
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    return f"<h3>Prediction: {result}</h3><br><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
