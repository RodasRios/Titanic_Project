from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import LabelEncoder


def evaluate_model(model, X_train, y_train, X_test, y_test, parametros):

        y_pred = model.predict(X_test)
        print("Accuracy of de model optimized: ", accuracy_score(y_test, y_pred))

        rand_est = GridSearchCV(model, parametros, cv=5, scoring= 'accuracy').fit(X_train,y_train.values.ravel())

        y_pred = rand_est.predict(X_test)


        print(rand_est.best_estimator_)
        print(rand_est.best_params_)
        print(rand_est.best_score_)
        print("Accuracy of de model: ", accuracy_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        ConfusionMatrixDisplay(confusion_matrix=cm).plot()

        plt.show()

        model_opt = RandomForestClassifier(bootstrap= True, criterion='entropy', max_depth=8, max_features='log2', n_estimators=104)
        model_opt.fit(X_train, y_train)

        y_pred = model_opt.predict(X_test)
        print("Accuracy of de model optimized: ", accuracy_score(y_test, y_pred))
