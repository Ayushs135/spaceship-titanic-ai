import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

train_df = pd.read_csv("C:\\Users\\ayush\\OneDrive\\Desktop\\vs\\Git\\AI_hackathon\\train.csv")
test_df = pd.read_csv("C:\\Users\\ayush\\OneDrive\\Desktop\\vs\\Git\\AI_hackathon\\test.csv")

train_df[['Deck', 'Num', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)


train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

train_df['HomePlanet'].fillna(train_df['HomePlanet'].mode()[0], inplace=True)
test_df['HomePlanet'].fillna(test_df['HomePlanet'].mode()[0], inplace=True)

train_df['CryoSleep'].fillna(False, inplace=True)
test_df['CryoSleep'].fillna(False, inplace=True)

train_df['VIP'].fillna(False, inplace=True)
test_df['VIP'].fillna(False, inplace=True)

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


train_df = pd.get_dummies(train_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
test_df = pd.get_dummies(test_df, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
combined_df = pd.concat([train_df, test_df], axis=0)

combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


X = train_df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Transported'])
y = train_df['Transported']



missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0

X_test = test_df.drop(columns=['PassengerId', 'Name', 'Cabin'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=41)


model = GradientBoostingClassifier(random_state=41)
model.fit(X_train, y_train)
param_dist = {'n_estimators': randint(50, 200),
                  'max_depth': randint(5, 20),
                  'min_samples_split': randint(2, 10),
                  'min_samples_leaf': randint(1, 5)}

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')
X_test = test_df.drop(columns=['PassengerId', 'Name', 'Cabin'])


test_pred = model.predict(X_test)


submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': test_pred})
submission.to_csv('Final_submission.csv', index=False)