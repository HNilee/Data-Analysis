import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

pf = pd.read_csv("PurchaseFinal2016.csv")
pf["ReceivingDate"] = pd.to_datetime(pf["ReceivingDate"], errors='coerce')
pf["PODate"] = pd.to_datetime(pf["PODate"], errors='coerce')

pf = pf.dropna(subset=["ReceivingDate", "PODate"])

daily = pf.groupby(pf["ReceivingDate"].dt.date).sum(numeric_only=True).reset_index()

daily["ReceivingDate"] = pd.to_datetime(daily["ReceivingDate"])

daily["Day"] = daily["ReceivingDate"].dt.day
daily["Month"] = daily["ReceivingDate"].dt.month
daily["Year"] = daily["ReceivingDate"].dt.year
daily["Weekday"] = pd.to_datetime(daily["ReceivingDate"]).dt.weekday

X = daily[["Day", "Month", "Year", "Weekday"]]
y = daily["Dollars"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

score = rfr.score(X_test, y_test)
print("R2 Score:", score)

joblib.dump(rfr, "purchase.sav")
print("Model saved as purchase.sav")
