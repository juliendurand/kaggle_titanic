import numpy as np
import pandas as pd

# Import the Data into the Script
train = pd.read_csv("./input/train.csv", dtype={
                    "Survived": np.bool_,
                    "Pclass": np.int32,
                    "Age": np.float64,
                    "Fare": np.float64},
                    )
test = pd.read_csv("./input/test.csv", dtype={
                   "Age": np.float64,
                   "Fare": np.float64},
                   )

# Create the submission file with passengerIDs from the test file
submission = pd.DataFrame({"PassengerId": test['PassengerId'],
                          "Survived": pd.Series(dtype='int32')})

survived = train['Survived']
sex = train['Sex']
age = train['Age']
fare = train['Fare']
cl = train['Pclass']
sib = train['SibSp']
size = len(sex)

correct = 0
total = 0

threshold = 12
for i in range(0, size):
    if sex[i] == 'male' and age[i] < threshold and sib[i] < 3:
        total += 1
        if survived[i]:
            correct += 1

print(correct/total)
print(correct)
print(total)

correct = 0
total = 0
for i in range(0, size):
    if sex[i] == 'male' and age[i] < threshold and sib[i] > 3:
        total += 1
        if survived[i]:
            correct += 1

print(correct/total)
print(correct)
print(total)


def classify(data):
    results = []
    size = len(data)
    sex = data['Sex']
    age = data['Age']
    fare = data['Fare']
    sib = data['SibSp']
    cl = data['Pclass']
    emb = data['Embarked']

    for i in range(0, size):
        survived = 0
        if sex[i] == 'female':
            if cl[i] < 3:
                survived = 1
            elif fare[i] < 23:
                if emb[i] != 'S':
                    survived = 1
                elif 11 < fare[i] <= 18:
                    survived = 1
        else:  # male
            if age[i] < 12 and sib[i] < 3:
                survived = 1
        results.append(survived)
    return results


def score():
    size = len(train)
    results = classify(train)
    total = 0
    valid = 0
    for i in range(0, size):
        total += 1
        if results[i] == train['Survived'][i]:
            valid += 1
    s = valid / total
    print('Score on training set : ' + str(s))

score()

submission['Survived'] = classify(test)

# Create final submission file
submission.to_csv("submission.csv", index=False)
