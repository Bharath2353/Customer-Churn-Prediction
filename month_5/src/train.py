from sklearn.model_selection import train_test_split
from src.preprocess import load_data, preprocess
from src.model import build_model
import pickle

df = load_data("data/customer_churn.csv")

X, y, columns, scaler = preprocess(df)   # ✅ FIXED

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("model.keras")

pickle.dump(columns, open("columns.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))   # ✅ IMPORTANT