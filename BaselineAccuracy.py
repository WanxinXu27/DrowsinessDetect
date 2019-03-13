import pandas as pd
import numpy as np
from OneHotEncoding import process_nominal_features
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
groundTruth = pd.read_csv('./groundtruth/ground_truth.csv')
baseline = pd.read_csv('./output/baseline_output.csv')
error = []
total = 0
correct = 0
for i in range(len(groundTruth)):
    if str(groundTruth['GroundTruth'][i]) != 'nan':
        total += 1
        if groundTruth['GroundTruth'][i] == baseline['IsDrowsy'][i]:
            correct += 1
        else:
            error.append(i)

print(baseline.iloc[error])
baseline.iloc[error].to_csv('./output/Error_baseline.csv', index=False)
with open('./output/Accuracy_baseline.txt', 'w') as f:
    f.write(str(correct) + ' / ' + str(total))

df = pd.merge(groundTruth, baseline, on='Video')
df = df.dropna()

print("accuracy: ",correct/total )
truth = df.loc[:, 'GroundTruth'].values.reshape(len(df), -1)
truth = process_nominal_features(truth)
prediction = df.loc[:, 'IsDrowsy'].values.reshape(len(df), -1)
prediction = process_nominal_features(prediction)
f1 = f1_score(truth, prediction, average='weighted')
print("f1 score: ", f1)

auc = roc_auc_score(truth, prediction, average='weighted')
print("auc scoreL ", auc)