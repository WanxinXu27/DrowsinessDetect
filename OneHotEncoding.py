import pandas as pd
from sklearn import preprocessing
import numpy as np


def onehot_feature(df):
    f = np.ndarray((len(df), 3))
    for i in range(2, 5):
        le = preprocessing.LabelEncoder()
        f[:, i-2] = le.fit_transform(df.iloc[:, i].values)
    # f1 = np.ndarray((len(df),1))
    # f1[:, 0] = df.iloc[:, 1].values
    f = np.hstack((df.iloc[:, 1].values.reshape(-1,1), f))
    print(f)

    enc = preprocessing.OneHotEncoder()
    f_v = enc.fit_transform(f[:, 0: 2]).toarray()
    f_v = np.hstack((f_v, f[:, 2].reshape(-1,1)))
    print(f_v)
    return f_v, f[:, -1].reshape(-1,1)


def onehot_feature_multiclass(df):
    f = np.ndarray((len(df), 5))
    for i in range(2, 7):
        le = preprocessing.LabelEncoder()
        f[:, i-2] = le.fit_transform(df.iloc[:, i].values)
    # f1 = np.ndarray((len(df),1))
    # f1[:, 0] = df.iloc[:, 1].values
    f = np.hstack((df.iloc[:, 1].values.reshape(-1,1), f))
    print(f)

    enc = preprocessing.OneHotEncoder()
    f_v = enc.fit_transform(f[:, 0: 2]).toarray()
    f_v = np.hstack((f_v, f[:, 2:5].reshape(-1,3)))
    print(f_v)
    return f_v, f[:, -1].reshape(-1,1)


def feature_processing(df, numericfeature, nominalfeature, stringfeature):
    # classlabel = df.loc[:, 'class'].values.reshape(len(df),-1)
    # numericfeature = df.loc[:, numericfeature].values.reshape(len(df),-1)
    # nominalfeature = df.loc[:, nominalfeature].values.reshape(len(df),-1)
    # stringfeature = df.loc[:, stringfeature].values.reshape(len(df),-1)
    # feature = np.hstack((process_numeric_features(numericfeature),process_string_features(stringfeature),
    #                      process_nominal_features(nominalfeature) ))
    # label = process_nominal_features(classlabel)

    if numericfeature:
        numericfeature = scale_numeric_features(df.loc[:, numericfeature].values.reshape(len(df),-1))
        # print(numericfeature.shape)
        # numericfeature = df.loc[:, numericfeature].values.reshape(len(df), -1)
    else:
        numericfeature = np.zeros((len(df),1))
    if nominalfeature:
        nominalfeature = process_nominal_features(df.loc[:, nominalfeature].values.reshape(len(df),-1))
    else:
        nominalfeature = np.zeros((len(df),1))
    if stringfeature:
        stringfeature = process_string_features(df.loc[:, stringfeature].values.reshape(len(df),-1))
    else:
        stringfeature = np.zeros((len(df),1))
    feature = np.hstack((numericfeature,stringfeature,nominalfeature))

    classlabel = df.loc[:, 'GroundTruth'].values.reshape(len(df), -1)
    label = process_nominal_features(classlabel)
    return feature, label


def process_numeric_features(numericFeature):
    enc = preprocessing.OneHotEncoder()
    f = enc.fit_transform(numericFeature).toarray()
    return f


def scale_numeric_features(numericFeature):
    scaler = preprocessing.StandardScaler()
    f = scaler.fit_transform(numericFeature)
    return f



def process_nominal_features(nominalFeature):
    nominalf = np.zeros(nominalFeature.shape)
    for i in range(nominalFeature.shape[1]):
        le = preprocessing.LabelEncoder()
        nominalf[:, i] = le.fit_transform(nominalFeature[:,i])
    # print(nominalf)
    return nominalf


def process_string_features(stringFeature):
    stringf = process_nominal_features(stringFeature)
    enc = preprocessing.OneHotEncoder()
    return enc.fit_transform(stringf).toarray()


def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1, cv=10)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    print(grid_search.best_score_)
    return model


if __name__ == '__main__':
    path = './output/eyeFeatures.csv'
    df = pd.read_csv(path, index_col=False)  # input
    print(df)
