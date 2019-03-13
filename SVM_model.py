from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import OneHotEncoding
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2


def SVM_result():
    df_eye_feature = pd.read_csv('./output/eyeFeatures.csv')
    df_mouth_feature =  pd.read_csv('./output/mouthFeatures-0307-2.csv')
    df_groundtruth = pd.read_csv('./groundtruth/ground_truth.csv')
    df_train = pd.merge(df_eye_feature, df_mouth_feature, on='Video')
    df_train = pd.merge(df_train, df_groundtruth, on='Video')
    df_train.dropna(inplace=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print(df_train)
    df_train.to_csv('./output/training_set.csv', index=False)
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
    print(len(train_y))

    # # do CV GridSearch, test on test set
    # model = OneHotEncoding.svm_cross_validation(train_x,train_y.reshape(-1,))
    # print(np.average(cross_val_score(model, train_x, train_y.reshape(-1, ), cv=10)))

    # # Cross Validation on training dataset
    m2 = svm.SVC(C=1000, gamma=0.001)
    print(np.average(cross_val_score(m2,train_x,train_y.reshape(-1,), cv=10)))
    predicted = cross_val_predict(m2, train_x, train_y.reshape(-1, ), cv=10)

    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print(metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    print(len(df_train.iloc[error_index]))
    error_df = df_train.iloc[error_index].sort_values(by=['Video'])
    error_df.to_csv('./output/Error_svm.csv', index=False)


if __name__ == '__main__':
    SVM_result()
