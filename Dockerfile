FROM python:3.11

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the model file is generated before running the app
RUN python -c "import joblib; from sklearn.datasets import load_iris; from sklearn.ensemble import RandomForestClassifier; iris=load_iris(); X,y = iris.data, iris.target; model=RandomForestClassifier(); model.fit(X,y); joblib.dump(model, 'app/model.joblib')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
