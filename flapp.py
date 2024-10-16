from flask import Flask, request, render_template
import joblib

model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tf-idfvector.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        mail_text = request.form['mail_text']                
        mail_tfidf = vectorizer.transform([mail_text])                
        prediction = model.predict(mail_tfidf)               
        if prediction[0] == 1:
            result = "SPAM ⚠️"
        else:
            result = "HAM!"
        
        return render_template('result.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
