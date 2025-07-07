# Travel_Insuranse_EDA_Predictive_Modeling
Data analysis and predictive modeling of Travel insurance data which is gathered at India
This project presents a comprehensive exploratory data analysis (EDA) and predictive modeling workflow for a travel insurance dataset. The primary objective is to identify the key factors that influence customers to purchase travel insurance and to develop predictive models that classify potential buyers with high precision.

📊 Project Overview
The notebook/script is organized into the following core sections:

1️⃣ Introduction
Outlines the project objectives and a breakdown of the analytical process.

2️⃣ Data Loading & Preprocessing
Loads and cleans the travel insurance dataset.

Checks for missing values and describes dataset features.

Categorizes variables into categorical, ordinal, and continuous.

3️⃣ Exploratory Data Analysis (EDA)
Visualizes distributions of the target variable (TravelInsurance).

Examines relationships between predictors and the target using:

Countplots

Histograms

Pairplots

Correlation matrices

4️⃣ Feature Engineering
Encodes categorical variables.

Normalizes continuous variables.

Converts binary features into boolean types.

Conducts minor data transformations to improve modeling readiness.

5️⃣ Predictive Modeling
Trains several machine learning classification models:

Logistic Regression

Random Forest

K-Nearest Neighbors

Naive Bayes

Support Vector Machine (SVM)

Decision Tree

Gradient Boosting

AdaBoost

XGBoost

Automated TPOT AutoML classifier

Model evaluation was performed exclusively on macro precision scores to prioritize balanced predictive power across both classes.

📊 Key Findings
By feature importance from the top-performing models, the most significant factors predicting a travel insurance purchase were:

Annual Income (most important)

Family Members and Age (alternating in importance)

Commonly important features:

Ever Traveled Abroad

Frequent Flyer

Employment Type

📈 Model Performance Summary
Among all evaluated models, the four best performers based on macro precision were:

XGBoost Classifier – 0.8665 macro precision

Bagged Random Forest

AdaBoost

Random Forest

The XGBoost Classifier achieved the highest macro precision score, accurately predicting potential travel insurance buyers 86.65% of the time.

📌 Conclusion
This data science project successfully demonstrated the value of machine learning for predicting travel insurance purchase behavior. Key takeaways:

Annual income is the strongest predictor of travel insurance purchase.

Other relevant factors include family size, age, prior international travel experience, frequent flyer status, and employment sector.

Among tested models, XGBoost Classifier achieved the highest macro precision.

📊 Business Implications
The insights from this project offer actionable strategies for tour and travel companies:

Targeted Marketing: Focus promotions on customer groups with high annual incomes, larger families, frequent flyers, and those with prior international travel.

Sales Prioritization: Use the trained predictive model to identify and rank potential customers most likely to purchase travel insurance.

Improved Profitability: Data-driven targeting can lead to more efficient marketing campaigns, increased insurance sales, and higher overall profitability.

📈 Areas for Improvement
Future work could further enhance the predictive power and applicability of the model by:

Expanding the dataset with additional variables such as gender, which may reveal new insights.

Increasing sample diversity, as the current dataset includes only customers aged 25 to 35.

Advanced feature engineering techniques to better capture non-linear relationships or interactions between variables.

📦 Dependencies
The following Python libraries were used:

pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
tpot
📑 License
This project is released under the MIT License.
