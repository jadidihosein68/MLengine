import pandas as pd
from flask import Flask
from flask import jsonify
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 


app = Flask(__name__)
music_data = pd.read_csv('music.csv')

@app.route('/')
def index():
    return 'I am alive'

@app.route('/predict/gender/<int:gender>/age/<int:age>')
def show_predict(gender,age):
    try :
        model = joblib.load('music-recommaner.joblib')
    except :
        return ("call /trainModel")

    predictions = model.predict([[age,gender]])
    return jsonify(str(predictions))

#http://localhost:5000/predict/gender/1/age/32

@app.route('/trainModel')
def update_model():
    X = music_data.drop(columns=['genre']) #clean data
    y = music_data['genre'] #clean data
    model =DecisionTreeClassifier()
    model.fit(X,y) # train model 
    joblib.dump(model,'music-recommaner.joblib') # save model
    df = pd.read_csv('vgsales.csv')
    return jsonify(str(df.shape))



@app.route('/giveChart')
def give_chart():
    model = joblib.load('music-recommaner.joblib')
    tree.export_graphviz(model, out_file='music-recommaner.dot', 
                     feature_names=['age', 'gender'], 
                     class_names=sorted(y.unique()),
                    label='all', rounded = True , filled = True )
    return 'Export the dot file'

@app.route('/testModel/testSize/<float:testSize>')
def test_model(testSize):  
    X = music_data.drop(columns=['genre'])
    y = music_data['genre']
    X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = testSize)
    model =DecisionTreeClassifier()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    score = accuracy_score (y_test, predictions)
    return jsonify(str(score))
    #Sample : http://localhost:5000/testModel/testSize/0.2


    #set FLASK_APP=hello.py
    #flask run