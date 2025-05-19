# FinalCapstoneProject: A model to predict Churn 
Capstone Project 24.1: Final Report

**Author:** Christian Daniel Guerra Maraza

The present document will show a Business presentation followed by a Technical Report for the analysis and modeling

#### Problem Statement
The company is a medium-size bank that operates across 3 countries: Spain, France and Germany; its customers are mainly adult people between 30 and 40 years old and offers 4 products. Tha Bank has an issue with its churning customers in the last years and want to know how to identify customers that are going to cancel their accounts to design retention strategies. The bank does not have enough information that can be used for modeling due to data protection authorization from its customers, so the challenge is to use the available data to construct a predictive model to identify what customer are likely to churn.
The potential benefits are that the bank could deploy actions before the customer already churned, and maintain the relationship with them during time.

#### Available data:

<img width="410" alt="Screenshot 2025-05-18 at 6 05 31 pm" src="https://github.com/user-attachments/assets/2f156ed1-af13-4e0d-82ad-dd773db574da" />

# 1. Business Presentation

#### 1.1 Portfolio insights
<img width="1074" alt="Screenshot 2025-05-18 at 6 04 00 pm" src="https://github.com/user-attachments/assets/2c53854b-4253-48e7-850a-a55968ae0f64" />

- Comparing age distributions between those who churned and stayed, it could be say that **churners are older** than the other group.
- Number of Products seems to be important to identify churners. Customers with more than 2 products are more likely to churn.
- Balance distribution tell us that churners have a higher balance, so it is **CRITICAL** to deploy retention actions in the Bank.
- **20%** of the customers are churners
- A strong negative correlation between Number of Products and Balance is detected **(-0.38)**, it means that customers that have more products have less balance. Regularly this relationship should be positive when customers are loyal.
-----
#### 1.2 Modeling
<img width="716" alt="Screenshot 2025-05-18 at 6 39 54 pm" src="https://github.com/user-attachments/assets/61ee451b-b5b4-442c-a295-d8dec72de298" />

- The winning model is **RandomForest**, as it achieved the highest F1-score **(0.7087)** and also has the **highest Accuracy and Precision** among all models
- The model is quite reliable when it predicts that a customer will churn (high precision) however, it misses some actual churners (recall is not very high). Overall, it is a useful and solid model for decision-making, such as launching targeted retention campaigns.
- The most important features that help to classify churner customers are: **Age** and **Number of Products**, the rest of features have similar and low values of importance.
  
* **Note:** The best model is selected based on the **F1-score** because provides a good balance between Precision and Recall when there is class imbalance (few churners). High precision = few false alarms (not many customers wrongly predicted as churn) and High recall = the model doesn't miss many real churners.*
-----
#### 1.3 Analytical business recommendations
<img width="1529" alt="Screenshot 2025-05-18 at 5 28 01 pm" src="https://github.com/user-attachments/assets/92dafee7-c8cc-44bd-98fe-def42cb8f146" />

- The winning Random Forest Model was trained with **100 decision trees**, so to have a better business understanding of the model, the most recurrent rules were ranked obtaining the chart above.
- There are **2 rules** that have more ocurrencies in the decision tress, those are:
  - (Geography = 'France') AND (NumOfProducts > 2)
  - (Age > 42) AND (NumOfProducts > 2)
- It can be said that churner customers are those that are older than **42 years old** and have **3 or more products.** **So, intense cross-selling actions must be stopped to those customers with 2 products and age > 42.**
- In France, age does not matter to identify churners only the number of products.
- Customers with a **"poor"** credit score are churners, but it is OK if they decide to cancel their accounts due to the credit risk level that they have.
-----

# Technical Report

#### Research Question
How can I identify the customers that are going to churn or are more likely to churn the Bank?

#### Data Sources
https://www.kaggle.com/api/v1/datasets/download/saurabhbadole/bank-customer-churn-prediction-dataset?dataset_version_number=2

#### Methodology
The present EDA will be use to generate Logistic regression, SVM and Random Forest models.
