import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def preprocess(dataframe, is_train):
    col_keys = dataframe.keys()
    '''
    ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    '''
    labels = None
    if(is_train):
        labels = dataframe['Survived']
        dataframe = dataframe.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
    else:
        dataframe = dataframe.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    col_keys = dataframe.keys()

    # male = 1, female = 2, unknown = 0
    dataframe['Sex'] = dataframe['Sex'].transform(
        lambda s: 1 if s=="male" else 2 if s=="female" else 0)

    # absent ages are -1.
    dataframe['Age'] = dataframe['Age'].transform(
        lambda x: -1.0 if np.isnan(x) else x)

    # absent fare = -1
    dataframe['Fare'] = dataframe['Fare'].transform(
        lambda x: -1.0 if np.isnan(x) else x)

    # Southhampton = 1, Queenstown = 2, Cherbourg = 3, unknown = 0
    dataframe['Embarked'] = dataframe['Embarked'].transform(
        lambda s: 1 if s=='S' else 2 if s=='Q' else 3 if s=='C' else 0)

    processed_data = dataframe.to_numpy(dtype=np.float32)
    if(is_train):
        return processed_data, labels.to_numpy(dtype=np.int32)
    else:
        return processed_data

def _knn(X_train, labels, X_test, test_labels, nn_ct):
    model = NearestNeighbors(n_neighbors=nn_ct, algorithm='ball_tree').fit(X_train)
    survived_pred = labels[model.kneighbors(X_test,return_distance=False)]
    gen_labels = np.sum(survived_pred, axis=1)>nn_ct//2
    print(gen_labels)
    print(test_labels)
    # add the final error calc.
    return 1-np.count_nonzero(test_labels==gen_labels)/gen_labels.shape[0]

def run():
    train_frame = pd.read_csv("train.csv")
    test_slice = train_frame.iloc[791:890]
    train_slice = train_frame.iloc[0:790]
    
    test_frame = pd.read_csv("test.csv")
    
    train_data, labels = preprocess(train_slice, True)
    train_slice_data, slice_labels = preprocess(test_slice, True)
    test_data = preprocess(test_frame, False)

    print(_knn(train_data, labels, train_slice_data, slice_labels, 5))
    
run()


