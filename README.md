# Travel_Insuranse_EDA_Predictive_Modeling
Data analysis and predictive modeling of Travel insurance data which is gathered at India
This project conducts a comprehensive exploratory data analysis (EDA) and applies predictive machine learning techniques to a travel insurance dataset. The primary goal is to identify the most significant factors that influence customers to purchase travel insurance and to develop predictive models that can effectively classify potential insurance buyers.

📊 Project Overview
The notebook (now converted to a .py script) is organized into the following key sections:

1️⃣ Introduction
Outlines the project objectives, motivations, and a high-level breakdown of the workflow.

2️⃣ Data Loading & Preprocessing
Loads the travel insurance dataset.

Performs initial data cleaning (dropping unnecessary columns, checking for missing values).

Describes dataset features, categorizing them as categorical, ordinal, or continuous.

3️⃣ Exploratory Data Analysis (EDA)
Visualizes the distribution of the target variable (TravelInsurance).

Analyzes categorical, ordinal, and continuous features using:

Countplots

Histograms

Pairplots

Correlation matrices

Examines relationships between features and the target variable to uncover trends and patterns.

4️⃣ Feature Engineering
Encodes categorical features.

Normalizes continuous variables.

Identifies and handles outliers.

Creates new features where beneficial for modeling.

5️⃣ Predictive Modeling
Applies a suite of machine learning classification algorithms:

Logistic Regression

Random Forest

K-Nearest Neighbors

Naive Bayes

Support Vector Machines (SVM)

Decision Tree

Gradient Boosting

AdaBoost

XGBoost

Automated TPOT AutoML classifier

Evaluates models using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC

6️⃣ Conclusion & Recommendations
Summarizes findings, compares model performances, and discusses actionable insights for travel agencies to optimize insurance marketing strategies.

📁 Dataset Description
Source: TravelInsurancePrediction.csv
Key Variables:

Age

Employment Type

GraduateOrNot

AnnualIncome

FamilyMembers

ChronicDiseases

FrequentFlyer

EverTravelledAbroad

TravelInsurance (target)

📦 Dependencies
The project uses the following Python libraries:

bash
Copy
Edit
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
tpot
📌 How to Run
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Execute the script or open the notebook in Jupyter.

Follow through the analysis and model evaluations.

📈 Results Summary
The XGBoost and Random Forest Classifiers yielded the highest accuracy and balanced precision-recall scores.

Key factors influencing travel insurance purchase include:

Annual income

Age

Employment sector

Frequency of travel

Prior international travel experience

📑 License
This project is released under the MIT License.
