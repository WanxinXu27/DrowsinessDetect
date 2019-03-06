import pandas as pd
import numpy as np

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
