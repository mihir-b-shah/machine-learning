import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

'''
import warnings
warnings.filterwarnings('ignore')
'''

def preprocess(dataframe, is_train):
    col_keys = dataframe.keys()
    '''
    ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    '''
    labels = None
    ids = dataframe['PassengerId']
    
    if(is_train):
        labels = dataframe['Survived']
        dataframe = dataframe.drop(columns=['PassengerId', 'Survived', 'Name',
                                            'Ticket', 'Cabin'])
    else:
        dataframe = dataframe.drop(columns=['PassengerId', 'Name', 'Ticket',
                                            'Cabin'])

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
        return processed_data, ids

def error_rate(gen_labels, true_labels):
    return 1-np.count_nonzero(true_labels==gen_labels)/gen_labels.shape[0]

def _knn(X_train, labels, X_test, test_labels, X_real, nn_ct=5):
    #train the model
    model = NearestNeighbors(n_neighbors=nn_ct, algorithm='ball_tree').fit(X_train)

    #train set error
    survived_pred_tr = labels[model.kneighbors(X_train,return_distance=False)]
    gen_labels_tr = np.sum(survived_pred_tr, axis=1)>nn_ct//2

    #on the faux test set
    survived_pred = labels[model.kneighbors(X_test,return_distance=False)]
    gen_labels = np.sum(survived_pred, axis=1)>nn_ct//2

    #on real test set
    real_labels = labels[model.kneighbors(X_real,return_distance=False)]
    
    # add the final error calc.
    return real_labels, error_rate(gen_labels, test_labels)

def _svm(X_train, labels, X_test, test_labels, X_real, gamma=1, tolerance=1e-4,
         iter_ct=10000):
    model = LinearSVC(random_state=0, max_iter=iter_ct, C=1/gamma, tol=tolerance)
    model.fit(X_train, labels)

    # train set error
    gen_labels_tr = model.predict(X_train)

    # on the faux test set
    gen_labels = model.predict(X_test)

    #on real
    real_labels = model.predict(X_real)

    return real_labels, error_rate(gen_labels, test_labels)

def run():
    train_frame = pd.read_csv("train.csv")
    test_slice = train_frame.iloc[train_frame.shape[0]-100:train_frame.shape[0]]
    train_slice = train_frame.iloc[0:train_frame.shape[0]-100]

    train_data, labels= preprocess(train_slice, True)
    train_slice_data, slice_labels = preprocess(test_slice, True)

    outputs = []
    test_frame = pd.read_csv("test.csv")
    test_data, ids = preprocess(test_frame, False)
    
    outputs.append(_knn(train_data, labels, train_slice_data, slice_labels,
                        test_data, 7))
    
    outputs.append(_svm(train_data, labels, train_slice_data,
                                    slice_labels, test_data,
                                    gamma=0.1, iter_ct = 50000))

    best = min(outputs, key=lambda t: t[1])
    np.savetxt("results.csv", np.concatenate(ids, best[0]), delimiter=',')
    
run()


