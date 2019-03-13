from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import OneHotEncoding
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import f1_score
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

def preprocess_features():
    df_train = pd.read_csv('./output/training_set.csv')
    print(df_train)
    feature, label = OneHotEncoding.feature_processing(df_train,
                                                       [
                                                        'BlinkRate',
                                                        'AvgClosureDegree',
                                                        'MaxClosureFrames',
                                                        'ValidDuration',
                                                        'Blinks',
                                                        'Yawns'
                                                        ],
                                                       [],[])
    print('Preprocessing done!')
    train_x = feature[:len(df_train), :]
    train_y = label[:len(df_train), :]

    return train_x, train_y, df_train


def SVM_result(train_x, train_y, df_train):
    # # do CV GridSearch, test on test set
    # model = OneHotEncoding.svm_cross_validation(train_x,train_y.reshape(-1,))
    # print(np.average(cross_val_score(model, train_x, train_y.reshape(-1, ), cv=10)))

    # # Cross Validation on training dataset
    m2 = svm.SVC(C=1000, gamma=0.001)
    # print(np.average(cross_val_score(m2,train_x,train_y.reshape(-1,), cv=10)))
    predicted = cross_val_predict(m2, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    print(len(df_train.iloc[error_index]),'\n')
    error_df = df_train.iloc[error_index].sort_values(by=['Video'])
    error_df.to_csv('./output/Error_svm.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc score: ", auc)


def random_forests_model(train_x, train_y, df_train):
    classifier = RandomForestClassifier(max_depth=10, random_state=0)
    predicted = cross_val_predict(classifier, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print("accuracy", metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    print(len(df_train.iloc[error_index]))
    error_df = df_train.iloc[error_index].sort_values(by=['Video'])
    error_df.to_csv('./output/Error_svm.csv', index=False)

    f1 = f1_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("f1 score: ", f1)

    auc = roc_auc_score(train_y.reshape(-1, ), predicted, average='weighted')
    print("auc scoreL ", auc)
    ############################################
    model = classifier.fit(train_x, train_y)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(train_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(train_x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(train_x.shape[1]), indices)
    plt.xlim([-1, train_x.shape[1]])
    plt.show()


if __name__ == '__main__':
    train_x, train_y, df_train = preprocess_features()
    SVM_result(train_x, train_y, df_train)
    # random_forests_model(train_x, train_y, df_train)
