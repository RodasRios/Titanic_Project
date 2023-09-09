import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data():
    data = pd.read_csv('D:/Python proj/Titanic Project/datasets/train.csv', sep=',', header=0)
    test_data = pd.read_csv('D:/Python proj/Titanic Project/datasets/test.csv', sep=',', header=0)
    combined = data.append(test_data)
    return data, test_data, combined

def process_age_title_feature(data, test_data, combined):

        data['Title'], test_data['Title'] = [df.Name.str.extract \
                (' ([A-Za-z]+)\.', expand=False) for df in [data, test_data]]

        data.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count'])

        TitleDict = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty", \
                    "Don": "Royalty", "Sir" : "Royalty","Dr": "Royalty","Rev": "Royalty", \
                    "Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs","Mr" : "Mr", \
                    "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}

        data['Title'], test_data['Title'] = [df.Title.map(TitleDict) for df in [data, test_data]]

        print(data.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count']))

        print(data.groupby(['Pclass','Sex','Title'])['Age'].agg({'mean', 'median', 'count'}))

        pd.unique(data['Title'])

        print(data.shape)

        combined=data.append(test_data)

        for df in [data, test_data, combined]:
            df['PeopleInTicket']=df['Ticket'].map(combined['Ticket'].value_counts())
            df['FarePerPerson']=df['Fare']/df['PeopleInTicket']

        for df in [data, test_data, combined]:
            df.loc[(df['Title']=='Miss') & (df.Parch!=0) & (df.PeopleInTicket>1), 'Title']="FemaleChild"

        data.loc[(data['Age'] <= 10)&(data['Title'] == "Miss")]

        data.loc[777, 'Title']="FemaleChild"

        combined=data.append(test_data)
        print(data.shape,test_data.shape,combined.shape)

        print(data.head())

        grp = data.groupby(['Pclass','Sex','Title'])['Age'].mean()
        grp = data.groupby(['Pclass','Sex','Title'])['Age'].mean().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

        def fill_age(x):
            return grp[(grp.Pclass==x.Pclass)&(grp.Sex==x.Sex)&(grp.Title==x.Title)]['Age'].values[0]
        ##Here 'x' is the row containing the missing age. We look up the row's Pclass
        ##Sex and Title against the lookup table as shown previously and return the Age
        ##Now we have to call this fill_age function for every missing row for test, train

        data['Age'], test_data['Age'] = [df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1) for df in [data, test_data]]

        print(data.info())

        """Family"""

        data['Fam_Size'] = np.where((data['SibSp']+data['Parch']) == 0 , 'Solo',
                                    np.where((data['SibSp']+data['Parch']) <= 3, 'Nuclear', 'Big'))
        
        return data, test_data, combined


def process_data(data):
       
        le = LabelEncoder()
        data = data.drop(['Cabin'],axis=1)
        data.dropna(inplace = True)
        y= data['Survived']
        X = data.drop(['PassengerId','Survived','Name'],axis=1)
        X.Pclass = le.fit_transform(X.Pclass)
        X.Embarked = le.fit_transform(X.Embarked)
        X.Title = le.fit_transform(X.Title)
        X.Sex = le.fit_transform(X.Sex)
        X.Ticket = le.fit_transform(X.Ticket)
        X = pd.get_dummies(X, columns = ['Fam_Size'])


        scaler = MinMaxScaler()
        data_pre = pd.concat([X,y],axis=1)
        data_scaled = scaler.fit_transform(data_pre)
        data_scaled = pd.DataFrame(data_scaled)
        data_scaled.columns = data_pre.columns
        y= data_scaled['Survived']
        X = data_scaled.drop(['Survived'],axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)


        return X_train, X_test, y_train, y_test

def process_test_data(test_data):
    test_data.loc[test_data['Parch'] == 9]
    test_data.groupby(['Pclass','Embarked'])['Fare'].agg(['mean','count'])
    test_data.loc[152, 'Fare'] = 13.913
    test_data.loc[152, 'FarePerPerson'] = 13.913

    test_data['Fam_Size'] = np.where((test_data['SibSp']+test_data['Parch']) == 0 , 'Solo',
                            np.where((test_data['SibSp']+test_data['Parch']) <= 3, 'Nuclear', 'Big'))
    test_data.loc[test_data.FarePerPerson.isnull()]

    test_data.loc[test_data.Title.isnull()]

    test_data.loc[414, 'Title'] = 'Royalty'

    le = LabelEncoder()
    test_data = test_data.drop(['Cabin'],axis=1)
    X_t = test_data.drop(['PassengerId','Name'],axis=1)
    X_t.Pclass = le.fit_transform(X_t.Pclass)
    X_t.Embarked = le.fit_transform(X_t.Embarked)
    X_t.Title = le.fit_transform(X_t.Title)
    X_t.Sex = le.fit_transform(X_t.Sex)
    X_t.Ticket = le.fit_transform(X_t.Ticket)
    X_t = pd.get_dummies(X_t, columns = ['Fam_Size'])
    X_test_test = X_t


    return X_test_test

