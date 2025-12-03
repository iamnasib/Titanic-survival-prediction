import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

def build_pipeline(cat_attribs):

    cat_pipeline= Pipeline([
        ('encoder',OrdinalEncoder()),
        ('imputer',SimpleImputer(strategy='most_frequent'))
    ])

    full_pipeline = ColumnTransformer([
        ('categorical',cat_pipeline,cat_attribs)
        ])

    return full_pipeline

def inference():
    print('Test Dataset Loading...')

    test_data = pd.read_csv('test.csv')

    print('Preprocessing Dataset...')
    X,y= preprocess(test_data)

    pipeline=joblib.load(PIPELINE_FILE)
    model=joblib.load(MODEL_FILE)
    print('Preprocessing Dataset...')
    preprocessed_df=pipeline.transform(X)
    
    test_data['Survival Prediction']=model.predict(preprocessed_df)
    test_data.to_csv('titanic-predictions.csv', index=False)

    print('Predictions saved to titanic-predictions.csv')

def preprocess(df):
    #COLUMNS  PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked

    X= df.copy()
    y= None
    if 'Survived' in df.columns: 
        X=df.drop('Survived', axis=1)
        y= df['Survived'].copy()

    # Dropping not required columns
    X.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

    X['Age']=X['Age'].fillna(-0.5)
    labels = ['Unknown', 'Baby', 'Child', 'Teenager',
          'Student', 'Young Adult', 'Adult', 'Senior']
    bins= [-1,0,5,12,18,25,35,60,np.inf]

    X['AgeGroup']=pd.cut(X['Age'],bins=bins,labels=labels)

    X.drop('Age', axis=1, inplace=True)

    return X,y

if not os.path.exists(MODEL_FILE):
    print('Training Beginning...')

    titanic_data = pd.read_csv('train.csv')

    print('Preprocessing Dataset...')
    X,y = preprocess(titanic_data)

    # print('Dataset Splitting...')
    # X_train,X_test, y_train, y_test = train_test_split(preprocessed_df,y, random_state=42, test_size=0.2)

    # pd.concat([X_test,y_test],axis=1).to_csv('test_data.csv', index=False)

    #Building Pipeline
    cat_attribs = X.select_dtypes(exclude=[np.number]).columns.tolist()

    pipeline=build_pipeline(cat_attribs)

    preprocessed_df=pipeline.fit_transform(X)

    print('Model Training...')
    model=RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(preprocessed_df, y)

    print('Saving Model and Pipeline...')
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print('Model and Pipeline saved.')

    inference()

else:
    inference()


