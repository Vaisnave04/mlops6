'''import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris=load_iris()
X,y = iris.data, iris.target
model=RandomForestClassifier()
model.fit(X,y)
joblib.dump(model, 'model.joblib')'''




from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os

# Load model using OS-independent path
model_path = os.path.join("app", "model.joblib")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)

# Class labels
class_names = np.array(['setosa', 'versicolor', 'virginica'])

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Iris model API"}

@app.post("/predict")
def predict(data: dict):
    features = data.get("features")
    
    if not features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request")
    
    try:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        class_name = class_names[prediction][0]
        return {"predicted_class": class_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
