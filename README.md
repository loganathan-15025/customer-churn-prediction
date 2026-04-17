**Customer Churn Intelligence System**

**Overview**
This project predicts customer churn using machine learning and provides insights to help businesses retain customers. It combines data preprocessing, model training, and an interactive Streamlit interface for real-time predictions.

**Problem Statement**
Customer churn leads to revenue loss. Identifying customers who are likely to leave helps businesses take proactive actions and improve retention strategies.

**Features**

* Predicts churn probability using trained ML models
* Provides key insights explaining churn behavior
* Suggests simple business actions based on risk level
* Clean and user-friendly Streamlit interface

**Tech Stack**

* Python
* Pandas
* Scikit-learn
* LightGBM
* Streamlit

**Approach**

**Data Preprocessing**
Handled missing values in TotalCharges
Removed unnecessary column customerID
Converted categorical variables using one-hot encoding

**Model Building**
Trained a baseline Random Forest model
Applied hyperparameter tuning using GridSearchCV
Trained LightGBM model for comparison

**Model Selection**
Random Forest achieved the best accuracy of approximately **79 percent**
LightGBM achieved approximately **78 percent**
Random Forest was selected as the final model

**Model Deployment**
Built an interactive UI using Streamlit
Enabled real-time prediction with user inputs
Displayed churn risk along with insights and recommendations

**Results**
Random Forest Accuracy approximately **79 percent**
LightGBM Accuracy approximately **78 percent**

**Key Insights**
Churn is mainly influenced by customer tenure, contract type, monthly charges, and payment method.
New customers and those with flexible contracts are more likely to churn.

**Project Structure**

customer-churn-prediction/
│
├── data/
│   └── churn.csv

├── src/
│   ├── train.py
│   ├── app.py

├── model.pkl
├── columns.pkl
├── requirements.txt
├── README.md

**How to Run**

Install dependencies
pip install pandas scikit-learn lightgbm streamlit

Train the model
cd src
python train.py

Run the application
streamlit run src/app.py

**Conclusion**
This project demonstrates a complete machine learning workflow from data preprocessing to deployment. It not only predicts churn but also provides useful insights to support business decisions.
