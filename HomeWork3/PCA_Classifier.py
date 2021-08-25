from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

lfw_people = fetch_lfw_people(min_faces_per_person=200, resize=0.4)
x = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=21)

pca = PCA(n_components=300, svd_solver='randomized',whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5 , 5e6 , 1e6],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1 , 0.5 ,1], }
clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator:" + str(clf.best_estimator_))
print("Best parametrs : "+str(clf.best_params_))
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf_neural = clf_neural.fit(X_train_pca, y_train)
y_pred = clf_neural.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
