import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

loaded_model = tf.keras.models.load_model('spam.h5')
cv = pickle.load(open('cv1.pkl','rb'))

@app.route('/')
def hello():
    return render_template('home.html')

##-------------------------------------------------

@app.route('/Spam', methods=['POST', 'GET'])
def prediction():
    return render_template('Spam.html')

##--------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = message

    # Process the new review text
    new_review = str(data)
    new_review = re.sub('[^a-zA-z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word)
                  for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)

    # Transform the new review text using the CountVectorizer from training
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_X_test = np.pad(
        new_X_test, [(0, 0), (0, 8672 - len(new_X_test[0]))], mode='constant')
    new_X_test = np.reshape(new_X_test, (new_X_test.shape[0], -1)) # Add this line to reshape input data
    # print(new_X_test)
  
    # Make predictions using the loaded model
    new_y_pred = loaded_model.predict(new_X_test)
    new_X_pred = np.where(new_y_pred > 0.5, 1, 0)
    print(new_X_pred)
    if new_X_pred[0][0] == 1:
        return render_template('result.html', prediction='Spam' ,post='spam')
    else:
        return render_template('result.html', prediction='Not a Spam',post='ham')

if __name__ == '__main__':
    app.run(debug=True)
