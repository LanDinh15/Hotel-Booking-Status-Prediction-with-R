# Hotel Booking Status Prediction (Classification)

This project predicts **hotel booking status** (Canceled vs Not_Canceled) using multiple classification models in **R**. It includes data exploration/visualization, train-test split, model training, ROC curves (for some models), and a final comparison of performance metrics.

## Models Implemented
- Decision Tree (rpart)
- Naive Bayes
- KNN (with feature scaling + k tuning)
- SVM (linear + radial) and KSVM (rbfdot / tanhdot / polydot)
- XGBoost (binary logistic)
- Random Forest
- Bagging (treebag via caret)
- Logistic Regression (full vs reduced model + ANOVA comparison)

## Key Result (Summary)
Based on the model comparison (Accuracy / F1 / Kappa / Classification Error), **Random Forest** achieved the best overall performance in this project.

> Note: Exact values may vary depending on dataset version, random seed, and tuning.

## Project Structure
├─ src/
│ └─ Hotel Booking Status Prediction.R
├─ slides/
│ └─ Hotel Booking Status Prediction.pptx
├─ data/
│ └─ Hotel Reservations.csv (not included in this repo)
├─ .gitignore
└─ README.md

