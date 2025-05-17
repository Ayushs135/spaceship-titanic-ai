
# 🚀 Spaceship Titanic - AI Hackathon Challenge

Welcome to the **Spaceship Titanic** repository! This project was created as part of an AI Hackathon challenge where the goal was to **predict which passengers were transported to an alternate dimension** after the spaceship collided with a spacetime anomaly.

## 🧩 Problem Statement

The interstellar passenger liner **Spaceship Titanic** was carrying over 13,000 passengers on its voyage to habitable exoplanets. After a collision with a spacetime anomaly, nearly half the passengers were mysteriously transported to another dimension. Using partially recovered data from the ship’s computer, our task is to build a machine learning model to predict whether each passenger was transported.

## 📁 Dataset Overview

The dataset includes features such as:

- `PassengerId`: Unique identifier
- `HomePlanet`: Planet the passenger is from
- `CryoSleep`: Whether the passenger was in cryosleep
- `Cabin`: Cabin assignment (Deck/Num/Side)
- `Destination`: Destination planet
- `Age`: Age of the passenger
- `VIP`: Whether the passenger is a VIP
- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`: Expenses across ship facilities
- `Name`: Passenger name
- `Transported`: **Target variable** (True if transported)

## 🧠 Approach

1. **Data Preprocessing**
   - Handled missing values using median (for numerical) and mode/defaults (for categorical).
   - Extracted features from `Cabin` (`Deck`, `Num`, `Side`).
   - Created dummy variables for categorical columns.
   - Extracted titles from `Name` (future enhancement).

2. **Modeling**
   - Used `GradientBoostingClassifier` from scikit-learn.
   - Performed a basic train/validation split.
   - Tuned hyperparameters using `RandomizedSearchCV` (included, optional).

3. **Evaluation**
   - Achieved competitive validation accuracy.
   - Final predictions exported to `Final_submission.csv`.

## 📦 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`

## 🏁 Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Ayushs135/spaceship-titanic-ai.git
   cd spaceship-titanic-ai
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn scipy
   ```

3. Place the dataset (`train.csv` and `test.csv`) in the working directory.

4. Run the model script:
   ```bash
   python model.py
   ```

5. The output `Final_submission.csv` will contain the predictions.

## 📊 Results

Achieved a validation accuracy of **~80%** (replace with your actual accuracy). Further improvements possible with feature engineering and model tuning.


## 📌 TODOs & Future Work

- Try deep learning models

## 🙌 Acknowledgements

Thanks to the organizers of the AI Hackathon for this fun and creative problem!
