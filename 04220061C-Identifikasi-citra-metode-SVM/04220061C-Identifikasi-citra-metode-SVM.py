from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, :12]
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
