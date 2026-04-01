from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.preprocessing import scale_data
import joblib


def train_model():
    X,y = load_iris(return_X_y=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_test,X_train = scale_data(X_test,X_train)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)

    joblib.dump(model,"model.pkl")

    return acc


if __name__ == "__main__":
    acc = train_model()
    print("accurayc",acc)