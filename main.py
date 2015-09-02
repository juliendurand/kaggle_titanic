import numpy as np
import pandas as pd

# Import the Data into the Script
train = pd.read_csv("./input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("./input/test.csv", dtype={"Age": np.float64}, )

# Create the submission file with passengerIDs from the test file
submission = pd.DataFrame({"PassengerId": test['PassengerId'],
                          "Survived": pd.Series(dtype='int32')})

sex = test['Sex']
age = test['Age']
size = len(sex)

# Fill the Data for the survived column, all females live and all males die
submission.Survived = [1 if sex[i] == 'female' or age[i] < 9.5 else 0 for i in range(0, size)]

# Create final submission file
submission.to_csv("submission.csv", index=False)
