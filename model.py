import joblib

def predict(data):
    clf = joblib.load("purchase.sav")
    return clf.predict(data)