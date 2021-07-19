import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 2
if __name__ == "__main__":
    with mlflow.start_run(run_name=str(neighbors) + '_neighbors'):
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy_train = metrics.accuracy_score(y_train,knn.predict(X_train))
        accuracy_test = metrics.accuracy_score(y_test,y_pred)
        print("Number of neighbours : %s" % neighbors)
        print("    train accuracy : %s" % accuracy_train)
        print("    test accuracy  : %s" %accuracy_test)
        mlflow.log_param("number_of_neighbors",neighbors)
        mlflow.log_metric("train_accuracy", accuracy_train)
        mlflow.log_metric("test_accuracy",accuracy_test)
