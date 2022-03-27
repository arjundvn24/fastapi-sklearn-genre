
# !pip install fastapi
# !pip install colabcode
# !pip install pyngrok
# !pip install --upgrade scikit-learn==0.20.3
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz 
import pydotplus
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
# !ngrok authtoken 26OKAP1NKdMpu3ebQkdImSSmrie_6tCp29v89ZV7RLXxXeNbJ
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# @app.post(Base)

df = pd.read_csv("https://raw.githubusercontent.com/kaustubhgupta/Technocolab-Final-Project/master/Data/cleaned.csv",  index_col=None)
df.head()

X=df.drop(['track_id','genre_top'],axis=1)
impX=["acousticness","danceability","energy","instrumentalness","liveness","speechiness","tempo","valence"]
y=df.genre_top

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)
tree=DecisionTreeClassifier(criterion="entropy",max_depth=3)
tree=tree.fit(X_train,Y_train)
treePrd=tree.predict(X_test)
dot_data=StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = impX,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
print("Accuracy:",metrics.accuracy_score(Y_test, treePrd))
# graph.write_png('image.png')
# Image(graph.create_png())

Pkl_Filename = "model_tree.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(tree, file)

# !pip install aiofiles
from fastapi.responses import FileResponse
app=FastAPI()
@app.get('/')
def index():
  return {"HEAD OF API"}
class Music(BaseModel):
    acousticness: float 
    danceability: float 
    energy: float 
    instrumentalness: float 
    liveness: float 
    speechiness: float 
    tempo: float 
    valence: float
    class Config:
        schema_extra = {
            "example": {
                "acousticness": 0.838816, 
                "danceability": 0.542950, 
                "energy": 0.669215,
                "instrumentalness": 0.000006,
                "liveness": 0.105610,
                "speechiness": 0.391221,
                "tempo": 111.894,
                "valence": 0.796073
            }
        }
@app.on_event("startup")
def load_model():
    global model
    model = pickle.load(open("model_tree.pkl", "rb"))
@app.post('/predict')
def get_music_category(data: Music):
    some_file_path = "image.png"  
    received = data.dict()
    acousticness = received['acousticness']
    danceability = received['danceability']
    energy = received['energy']
    instrumentalness = received['instrumentalness']
    liveness = received['liveness']
    speechiness = received['speechiness']
    tempo = received['tempo']
    valence = received['valence']
    pred_name = model.predict([[acousticness, danceability, energy,
                                instrumentalness, liveness, speechiness, tempo, valence]]).tolist()[0]
    return {'prediction': pred_name,
            'Accuracy':metrics.accuracy_score(Y_test, treePrd)}
    # return FileResponse(some_file_path)
