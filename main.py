import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


X,y = load_iris(return_X_y=True)
clf = RandomForestClassifier()
clf.fit(X, y)
joblib.dump(clf, "iris_model.pkl")