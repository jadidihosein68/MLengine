import pandas as pd
from flask import Flask
from flask import jsonify
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

app = Flask(__name__)
music_data = pd.read_csv('music.csv')
@app.route('/predict/gender/<int:gender>/age/<int:age>')
def show_predict(gender,age):
    model = joblib.load('music-recommaner.joblib')
    predictions = model.predict([[age,gender]])
    return jsonify(str(predictions))

#http://localhost:5000/predict/gender/1/age/32


@app.route('/updateModel')
def update_model():
    X = music_data.drop(columns=['genre']) #clean data
    y = music_data['genre'] #clean data
    model =DecisionTreeClassifier()
    model.fit(X,y) # train model 
    joblib.dump(model,'music-recommaner.joblib') # save model
    df = pd.read_csv('vgsales.csv')
    return jsonify(str(df.shape))

@app.route('/')
def index():
    return 'I am alive'

@app.route('/giveChart')
def give_chart():
    model = joblib.load('music-recommaner.joblib')
    tree.export_graphviz(model, out_file='music-recommaner.dot', 
                     feature_names=['age', 'gender'], 
                     class_names=sorted(y.unique()),
                    label='all', rounded = True , filled = True )
    return 'Export the dot file'

    #set FLASK_APP=hello.py
    #flask run