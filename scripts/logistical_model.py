from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

train_processed = pd.read_csv('D:/titanic_project/data/train_processed.csv')

X = train_processed.drop(columns=['PassengerId', 'Survived'])
y = train_processed['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

print("\nМатрица ошибок:") 
print(confusion_matrix(y_test, y_pred))

print("\nОтчёт по классификации:")
print(classification_report(y_test, y_pred))


coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
print("\nКоэффициенты модели (влияние признаков):")
print(coefficients.sort_values(by='Coefficient', ascending=False))

model.fit(X, y)

test_df = pd.read_csv('D:/titanic_project/data/train.csv')

test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp'] + 1
test_df['Age'] = test_df['Age'].fillna(test_df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df = test_df.drop(columns=['Name', 'Ticket', 'Cabin'])

missing_columns = set(X.columns) - set(test_df.columns)

for col in missing_columns:
    test_df[col] = 0

test_df = test_df[X.columns]

pred_test = model.predict(test_df)

submission = pd.DataFrame({
    'PassengerId': pd.read_csv('D:/titanic_project/data/train.csv')['PassengerId'],
    'Survived': pred_test
})

submission.to_csv('D:/titanic_project/data/submission.csv', index = False)