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

### 1.1 Portfolio insights
<img width="1074" alt="Screenshot 2025-05-18 at 6 04 00 pm" src="https://github.com/user-attachments/assets/2c53854b-4253-48e7-850a-a55968ae0f64" />

- Comparing age distributions between those who churned and stayed, it could be say that **churners are older** than the other group.
- Number of Products seems to be important to identify churners. Customers with more than 2 products are more likely to churn.
- Balance distribution tell us that churners have a higher balance, so it is **CRITICAL** to deploy retention actions in the Bank.
- **20%** of the customers are churners
- A strong negative correlation between Number of Products and Balance is detected **(-0.38)**, it means that customers that have more products have less balance. Regularly this relationship should be positive when customers are loyal.
-----
### 1.2 Model selection
<img width="716" alt="Screenshot 2025-05-18 at 6 39 54 pm" src="https://github.com/user-attachments/assets/61ee451b-b5b4-442c-a295-d8dec72de298" />

- The winning model is **RandomForest**, as it achieved the highest F1-score **(0.7087)** and also has the **highest Accuracy and Precision** among all models
- The model is quite reliable when it predicts that a customer will churn (high precision) however, it misses some actual churners (recall is not very high). Overall, it is a useful and solid model for decision-making, such as launching targeted retention campaigns.
- The most important features that help to classify churner customers are: **Age** and **Number of Products**, the rest of features have similar and low values of importance.  
<sub>**Note:** The best model is selected based on the **F1-score** because provides a good balance between Precision and Recall when there is class imbalance (few churners). High precision = few false alarms (not many customers wrongly predicted as churn) and High recall = the model doesn't miss many real churners.</sub>
-----
### 1.3 Analytical business recommendations
<img width="1529" alt="Screenshot 2025-05-18 at 5 28 01 pm" src="https://github.com/user-attachments/assets/92dafee7-c8cc-44bd-98fe-def42cb8f146" />

- The winning Random Forest Model was trained with **100 decision trees**, so to have a better business understanding of the model, the most recurrent rules were ranked obtaining the chart above.
- There are **2 rules** that have more ocurrencies in the decision tress, those are:
  - (Geography = 'France') AND (NumOfProducts > 2)
  - (Age > 42) AND (NumOfProducts > 2)
- It can be said that churner customers are those that are older than **42 years old** and have **3 or more products.** **So, intense cross-selling actions must be stopped to those customers with 2 products and age > 42.**
- In France, age does not matter to identify churners only the number of products.
- Customers with a **"poor"** credit score are churners, but it is OK if they decide to cancel their accounts due to the high credit risk level that they have.
-----

# 2. Technical Report

### 2.1 Model Outcomes or Predictions:
- **Type of learning:** Classification problem
- **Type of algorithm:** Supervised (there is a target variable that indicates if the customer churned)
- **Expected output of selected model:** The output will be a predicted classification for churner customers for the Bank


### 2.2 Data acquisition
Data provided by Kaggle website: https://www.kaggle.com/api/v1/datasets/download/saurabhbadole/bank-customer-churn-prediction-dataset?dataset_version_number=2

#### 2.2.1 Visualizations to assess data's potential

<img width="1729" alt="Screenshot 2025-05-18 at 3 10 19 pm" src="https://github.com/user-attachments/assets/d8927392-15e9-4787-9b44-2f8ea4869cd5" />

- Outliers are identified in CreditScore, Age and Number Of Products.
- The majority of customers have an acceptable Credit Scoring, mainly between 600 and 700.
- Customer age is commonly between 30 and 45 years old. The younger ones are 20 years old.
- The bank doesn't have many products, the maximum is 4 products but the average number of products is 1.
-----
 <img width="1516" alt="Screenshot 2025-05-18 at 3 10 31 pm" src="https://github.com/user-attachments/assets/c02caa4d-84e4-436f-9774-09953e375a26" />

 - There is no much older people in the Age distribution, so the bank is focused in the most productive people of the countries.
 - Seems to be an error in Balance variable, due to zero balance in an important number of accounts.
 - Product penetration is low, a few customers have more than 2 products.
-----
<img width="512" alt="Screenshot 2025-05-18 at 3 13 41 pm" src="https://github.com/user-attachments/assets/07496697-f3c0-4515-93ae-177a167b58cf" />

- This dataset has a 20% of churned customers (this is the target variable) so it could be said that the dataset is umbalanced.
-----
<img width="789" alt="Screenshot 2025-05-18 at 3 13 53 pm" src="https://github.com/user-attachments/assets/49d7d465-1db2-4cfd-bf1f-7e6dd1633c39" />

- Proportions of the Credit Score groups are very similar across countries. The Credit policy of the bank must be the same in all countries.
-----
<img width="1194" alt="Screenshot 2025-05-18 at 3 14 40 pm" src="https://github.com/user-attachments/assets/fa7b6fc5-013b-449d-963b-2e7a1e106bed" />

- There is more attrition as a ratio (2x) in Germany than the others countries.
- As a ratio, there is a similar level of cardholders accross countries.
- The % of active members is similar in all countries. However Germany has the lower activation ratio.
-----
<img width="1491" alt="Screenshot 2025-05-18 at 3 14 07 pm" src="https://github.com/user-attachments/assets/5a08c2b0-cf63-47a0-9f5b-8347abf3a668" />

- Numeric variables behave similarly among geographies. Only Germany reports a higher Balance distribution than the other countries.
-----
![corr_matriz](https://github.com/user-attachments/assets/ef325859-3cbf-4c69-a41e-25953ac260f1)

- Low correlations were identified between numeric variables.
- The highest correlation found is Number of products vs Balance (-0.38), followed by Number of products vs Credit Score (-0.11)

-----
![pairplot](https://github.com/user-attachments/assets/84914a21-f9cc-4682-be0d-38db0f3d399e)

- Seems to be low correlations accross variables.
- Number of products could be an important variable to classify churned customers.
- Customers who churned tend to be older than the others that stayed.
- Customers who churned have an important concentration in 1 products. If customers have 2 products or more they are less likely to churn.
-----

### 2.3 Data Preprocessing/Preparation:
#### 2.3.1 General Outliers treatment
To delete outliers the Z-score technique was used for better results. IQR and Standar Deviation techniques also were tested.
```
# Seleccionar solo columnas numéricas (excluyendo la target)
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Inicializar una máscara para mantener filas válidas
mask = pd.Series(True, index=df_cleaned.index)

# Aplicar Z-score e IQR por cada columna numérica
for col in num_cols:
    # Z-score
    z = np.abs(zscore(df_cleaned[col]))
    z_mask = z < 3  # umbral típico

    # IQR
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_mask = (df_cleaned[col] >= Q1 - 1.5 * IQR) & (df_cleaned[col] <= Q3 + 1.5 * IQR)

    # Mantener filas que son válidas para ambas técnicas
    mask &= z_mask & iqr_mask

# Aplicar máscara directamente sobre df_cleaned
original_shape = df_cleaned.shape
df_cleaned = df_cleaned[mask]

# Reporte de cuántas filas fueron eliminadas
print(f"Total de filas eliminadas por outliers: {original_shape[0] - df_cleaned.shape[0]}")
print(f"Dataset final sin outliers: {df_cleaned.shape}")
```
#### 2.3.2 Cleaning Balance variable
<img width="248" alt="Screenshot 2025-05-18 at 7 55 08 pm" src="https://github.com/user-attachments/assets/13ed0919-b60a-420e-9444-4d9a010c1d6e" />

As it was shown in EDA, Balance column has several zero values, so those rows were deleted and a new outliers treatment was applied. In this time, the IQR had better results.

```
# Numeric variable to consider
col = 'Balance'

# Initialize mask to keep valid rows
mask = pd.Series(True, index=df_cleaned.index)

# Z-score
z = np.abs(zscore(df_cleaned[col]))
z_mask = z < 3  # typical threshold

# IQR
Q1 = df_cleaned[col].quantile(0.25)
Q3 = df_cleaned[col].quantile(0.75)
IQR = Q3 - Q1
iqr_mask = (df_cleaned[col] >= Q1 - 1.5 * IQR) & (df_cleaned[col] <= Q3 + 1.5 * IQR)

# Keep rows valid in both techniques
mask &= z_mask & iqr_mask

# Apply mask directly to df_cleaned
original_shape = df_cleaned.shape
df_cleaned = df_cleaned[mask]

# Report number of rows removed
print(f"Total rows removed due to outliers in 'Balance': {original_shape[0] - df_cleaned.shape[0]}")
print(f"Final dataset shape without outliers: {df_cleaned.shape}")
```
**Variables after outliers treatment:**

<img width="1693" alt="Screenshot 2025-05-18 at 8 00 12 pm" src="https://github.com/user-attachments/assets/9677e7b2-eed9-4b1b-bcd9-57185e00610c" />

#### 2.3.3 Standadarization and encoding for categorical variables

- For Logistic Regression and SVM **standardization** of numercial features and **one-hot encoding** for categorical ones were applied.
- For Random Forest categorical features were codified using numbers or codes as follows:

<img width="222" alt="Screenshot 2025-05-18 at 4 31 52 pm" src="https://github.com/user-attachments/assets/6cdf4c35-3979-48d8-a9d2-1b26abd51e27" />

- To split the train and test samples **train_test_split()** function was used considering the default test_size parameter **(25%)** and enabling the **stratifed** option.

```
# Separar variables predictoras y variable objetivo
X = df_cleaned.drop(columns='Exited')
y = df_cleaned['Exited']

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

# Preprocesamiento para modelos sensibles: escalamiento y one-hot encoding
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Codificar variables categóricas como enteros para RandomForest
X_train_rfc = X_train.copy()
X_test_rfc = X_test.copy()
for col in categorical_features:
    X_train_rfc[col] = X_train_rfc[col].astype('category').cat.codes
    X_test_rfc[col] = X_test_rfc[col].astype('category').cat.codes
```
#### 2.3.4 Solving imbalance data

**SMOTE** technique was applied to balance the data training set for modeling

```
# Preprocesar y balancear para modelos sensibles
X_train_preprocessed = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)
X_test_preprocessed = preprocessor.transform(X_test)
```

<sub>For **X_test_preprocessed** the standardization and one-hot encoding were applied.</sub>

### 2.4 Modeling:

- **GridSearchCV()** was used to optimize the parameter configuration of the models.
- The 3 models were implemented using **pipelines** and many parameters were tested in the optimization.
- For **Logistic Regression** polynimial degrees of 1, 2 and 3 were tested in order to adjust better the model.
- For **SVM** regularization levels and both kernels were considered.
- For **RandomForest** were tested **100 and 200** trees and to control **overfiting**, the maximum deepth parameter was tested with 3, 5 and 8 levels.

 ```
# Models and grids definition
models = {
    'LogisticRegression': (
        Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        {
            'poly__degree': [1, 2, 3],  # Grados del polinomio
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__class_weight': ['balanced']
        }
    ),
    'SVM': (
        SVC(),
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    ),
    'RandomForest': (
        RandomForestClassifier(),
        {'n_estimators': [100, 200], 'max_depth': [3, 5, 8]}
    )
}

results = {}
best_estimators = {}

for model_name, (model, param_grid) in models.items():
    print(f"\nTraining {model_name}...")

    if model_name == 'RandomForest':
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train_rfc, y_train)
        y_pred = grid.predict(X_test_rfc)

    elif model_name == 'LogisticRegression':
        # El preprocesamiento + SMOTE se aplica fuera del pipeline
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train_balanced, y_train_balanced)
        y_pred = grid.predict(X_test_preprocessed)

    else:
        # SVM usa datos ya preprocesados y balanceados
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train_balanced, y_train_balanced)
        y_pred = grid.predict(X_test_preprocessed)

    report = classification_report(y_test, y_pred, output_dict=True)
    pos_label = list(report.keys())[-2]  # normalmente '1'

    results[model_name] = {
        'Best Params': grid.best_params_,
        'Test Accuracy': report['accuracy'],
        'Precision (class 1)': report[pos_label]['precision'],
        'Recall (class 1)': report[pos_label]['recall'],
        'F1-score (class 1)': report[pos_label]['f1-score']
    }

    best_estimators[model_name] = grid.best_estimator_
```

### 2.5 Model evaluation:

<img width="1168" alt="Screenshot 2025-05-18 at 5 36 02 pm" src="https://github.com/user-attachments/assets/6f43fa08-2089-454d-990f-1c8689d8430e" />

- The winning model is **RandomForest** with 100 decision trees and a maximum deepth of 8, as it achieved the highest F1-score **(0.7087)** and also has the **highest Accuracy and Precision** among all models.
- The model is quite reliable when it predicts that a customer will churn (high precision) however, it misses some actual churners (recall is not very high). Overall, it is a useful and solid model for decision-making, such as launching targeted retention campaigns.
  
<sub>**Note:** The best model is selected based on the **F1-score** because provides a good balance between Precision and Recall when there is class imbalance (few churners).

#### Feature importance

<img width="997" alt="Screenshot 2025-05-18 at 5 27 30 pm" src="https://github.com/user-attachments/assets/0b1e137a-820d-4800-9b45-5f80f799d707" />

- The most important features that help to classify churn customers are: Age and Number of Products.

```
# Mostrar importancias y árbol de RandomForest
rf_model = best_estimators['RandomForest']
importances = pd.Series(rf_model.feature_importances_, index=X_train_rfc.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\nTop 10 características más importantes en RandomForest:")
print(importances_sorted.head(10))

ax = importances_sorted.head(10).plot(kind='barh', figsize=(10, 6), title='Top 10 Feature Importances - RandomForest')
plt.gca().invert_yaxis()
for i, v in enumerate(importances_sorted.head(10)):
    ax.text(v + 0.001, i, f'{v:.2f}', va='center')
plt.tight_layout()
plt.show()
```
