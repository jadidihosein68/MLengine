import pandas as pd
from flask import Flask
from flask import jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load('music-recommaner.joblib')

@app.route('/predict/gender/<int:gender>/age/<int:age>')
def show_predict(gender,age):
    predictions = model.predict([[age,gender]])
    return jsonify(str(predictions))

#http://localhost:5000/predict/gender/1/age/32




@app.route('/')
def index():
    df = pd.read_csv('vgsales.csv')
    return jsonify(str(df.shape))

@app.route('/hello')
def hello():
    return 'Hello, World'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return 'Subpath %s' % subpath

  

@app.route('/about')
def about():
    return 'The about page'

    #set FLASK_APP=hello.py
    #flask run