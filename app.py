import pandas as pd # to manage dataframe 
from flask import Flask, request # use for rest 
from flask import jsonify # to return json 
from sklearn import tree # to cketch tree 
from sklearn.tree import DecisionTreeClassifier # to use clasification method 
from sklearn.externals import joblib # to export and import model 
from sklearn.model_selection import train_test_split # to train 
from sklearn.metrics import accuracy_score  # to test 
from flask_restplus import Resource, Api



app = Flask(__name__)

api = Api(app, version='1.0', title='Sample API',
    description='A sample API',
)


music_data = pd.read_csv('music.csv')




@api.route('/ping')
class ping(Resource):
    def get(self):
        """Ping application."""
        return {'ping':'I am alive'}


@api.route('/predict/gender/<int:gender>/age/<int:age>')
class predict(Resource):
    def get(self,gender,age):
        try :
            model = joblib.load('music-recommaner.joblib')
        except :
            return ("call /trainModel")

        predictions = model.predict([[age,gender]])
        return jsonify(str(predictions))

#http://localhost:5000/predict/gender/1/age/32



@api.route('/trainModel')
class trainModel(Resource):
    def get(self):
        """Train the model based the csv file."""
        X = music_data.drop(columns=['genre']) #clean data
        y = music_data['genre'] #clean data
        model =DecisionTreeClassifier()
        model.fit(X,y) # train model 
        joblib.dump(model,'music-recommaner.joblib') # save model
        df = pd.read_csv('vgsales.csv')
        return jsonify(str(df.shape))


@api.route('/giveChart')
class giveChart(Resource):
    def get(self):
        """Export the dot file."""
        model = joblib.load('music-recommaner.joblib')
        tree.export_graphviz(model, out_file='music-recommaner.dot', 
                        feature_names=['age', 'gender'], 
                        class_names=sorted(y.unique()),
                        label='all', rounded = True , filled = True )
        return 'Dot file exported successfully'

@api.route('/testModel/testSize/<float:testSize>')
class testModel(Resource):
    def get(self, testSize):  
        """Define test size."""
        X = music_data.drop(columns=['genre'])
        y = music_data['genre']
        X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = testSize)
        model =DecisionTreeClassifier()
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        score = accuracy_score (y_test, predictions)
        return jsonify(str(score))


@app.route('/secret')
def secret():
    return 'I am a secret api and not listed in swagger :)'


if __name__ == '__main__':
    app.run(debug=True)

#set FLASK_APP=hello.py
#flask run
