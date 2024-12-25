import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

path1="C:\\Users\\ayush\\OneDrive\\Desktop\\vs\\Git\\AI_hackathon\\train.csv"
train_df=pd.read_csv(path1)
#train_df.head(5)
path2="C:\\Users\\ayush\\OneDrive\\Desktop\\vs\\Git\\AI_hackathon\\test.csv"
test_df=pd.read_csv(path2)
#test_df.head(5)

# Data preprocessing
def preprocess_data(df):

    df['HomePlanet'].fillna('Unknown', inplace=True)
    df['CryoSleep'].fillna(False, inplace=True)
    df['Cabin'].fillna('Unknown/0/Unknown', inplace=True)
    df['Destination'].fillna('Unknown', inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['VIP'].fillna(False, inplace=True)
    df['RoomService'].fillna(0, inplace=True)
    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    df['HomePlanet'] = label_encoder.fit_transform(df['HomePlanet'])
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['Cabin'] = label_encoder.fit_transform(df['Cabin'])
    df['Destination'] = label_encoder.fit_transform(df['Destination'])
    df['VIP'] = df['VIP'].astype(int)

    df.drop(columns=['Name', 'PassengerId'], inplace=True)

    return df

train_df = preprocess_data(train_df)
test_df_ids = test_df['PassengerId']
test_df = preprocess_data(test_df)
X = train_df.drop(columns=['Transported'])
y = train_df['Transported'].astype(int)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

test_predictions = model.predict(test_df)

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': test_df_ids,
    'Transported': test_predictions
})
submission['Transported'] = submission['Transported'].astype(bool)
submission.to_csv('Base_submission.csv', index=False)