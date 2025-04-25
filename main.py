from src.preprocess import load_data
from src.model import train_model
from src.evaluate import evaluate

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(
        r"C:\Users\Archana Badri\PycharmProjects\iris-classifier\data\IRIS.csv")

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
