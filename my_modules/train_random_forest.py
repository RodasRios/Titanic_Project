from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
        
        

        model = RandomForestClassifier()

        parametros ={
                'bootstrap': [True],
                'n_estimators':range(100,800,50),
                'max_features': [ 'sqrt', 'log2'],
                'criterion': ['gini','entropy'],
                'max_depth': range(1,8),
                "min_samples_leaf" : [1, 5, 8],
                "min_samples_split" : [2, 4, 10]
                }
        '''
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        X_train = sc_x.fit_transform(X_train)
        X_test = sc_x.transform(X_test)
        '''

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)



        return model, parametros