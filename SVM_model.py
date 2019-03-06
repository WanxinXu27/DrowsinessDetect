from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
# import OneHotEncoding
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def SVM_result():
    name_train = 'Combined_shuffle'
    path_train = './Features/Feature_' + name_train + '.xlsx'
    df_train = pd.read_excel(path_train, 0)

    feature, label = OneHotEncoding.feature_processing(df_train,
                                                       ['length',
                                                        # 'common_f',
                                                        # 'O', 'U',
                                                        # 'B', 'C', 'D', 'F', 'G', 'J', 'M', 'P', 'V',
                                                        # # 'H',
                                                        # # 'L',
                                                        # # 'S',
                                                        # 'K', 'N', 'T', 'W',
                                                        # 'Q', 'R', 'X', 'Z',

                                                        # 'M+N', 'B+D+G', 'P+T+K', 'W+L+R', 'C+K+Q', 'S+Z', 'V+W', 'F+V',
                                                        # 'A+E+I+Y', 'A+E', 'A+O', 'I+Y', 'O+U+W',

                                                        ],
                                                       [
                                                           # 'VVV',
                                                           # 'CCC',
                                                           # 'CVC',
                                                           # 'VCV',
                                                           # 'common',
                                                           'in_lexicon',
                                                           # 'file',
                                                           # 'ER', 'AR','UR', 'OR', 'IR',
                                                           # 'NET',
                                                           # 'S_last',
                                                           # 'U_start'
                                                           # 'high_freq',
                                                           # 'high_freq_parenthesis'
                                                       ],
                                                       ['pattern',
                                                        # 'domain',
                                                        # 'start_letter',
                                                        # 'end_letter'
                                                        'common_stem_c',
                                                        ])
    print('Preprocessing done!')
    train_x = feature[:len(df_train), :]
    train_y = label[:len(df_train), :]
    print(len(train_y))

    # # do CV GridSearch, test on test set
    #     model = OneHotEncoding.svm_cross_validation(train_x,train_y.reshape(-1,))
    # print(np.average(cross_val_score(model, train_x, train_y.reshape(-1, ), cv=10)))
    # # # Cross Validation on training dataset
    m2 = svm.SVC(C=1000, gamma=0.001)
    # print(np.average(cross_val_score(m2,train_x,train_y.reshape(-1,), cv=10)))
    predicted = cross_val_predict(m2, train_x, train_y.reshape(-1, ), cv=10)
    error_index = np.where(np.equal(predicted, train_y.reshape(-1, )) == False)
    print(metrics.accuracy_score(train_y.reshape(-1, ), predicted))
    print(len(df_train.iloc[error_index]))
    df_train.iloc[error_index].to_excel('./Output/Error_' + name_train + '_SVM.xlsx', index=False)


if __name__ == '__main__':
    SVM_result()