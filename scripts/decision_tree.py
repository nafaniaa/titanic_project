from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_processed = pd.read_csv('D:/titanic_project/data/train_processed.csv')
X = train_processed.drop(columns=['PassengerId', 'Survived'])
y = train_processed['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# for depth in range(1,11):
#     dtc = DecisionTreeClassifier(max_depth=depth, random_state=42)
#     scores = cross_val_score(dtc, X, y, cv = 5)
#     print(f"Depth ={depth}, Mean accuracy = {np.mean(scores)}")
    #Depth =5, Mean accuracy = 0.817048521750047

model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy= accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)

conf_matr = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_matr)

model.fit(X, y)

test_df = pd.read_csv('D:/titanic_project/data/test.csv')

test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp'] + 1
test_df['Age'] = test_df['Age'].fillna(test_df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df = test_df.drop(columns=['Name', 'Ticket', 'Cabin'])

missing_columns = set(X.columns) - set(test_df.columns)

for col in missing_columns:
    test_df[col] = 0

test_df = test_df[X.columns]

test_pred = model.predict(test_df)

submission_dtm = pd.DataFrame({
    'PassengerId': pd.read_csv('D:/titanic_project/data/test.csv')['PassengerId'],
    'Survived': test_pred
})

submission_dtm.to_csv('D:/titanic_project/data/submission_dtm.csv', index = False)
