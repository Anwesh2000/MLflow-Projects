import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

mlflow.sklearn.autolog()
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 2
if __name__ == "__main__":
    mlflow.set_experiment(experiment_name='KNeighbors')
    with mlflow.start_run(run_name=str(neighbors) + '_neighbors'):
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
