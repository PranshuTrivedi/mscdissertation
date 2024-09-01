#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


engage_df=pd.read_csv('engage.csv')


# In[3]:


engage_df.head()


# In[4]:


# Drop the 'Unnamed: 0' column
engage_df = engage_df.drop(columns=['Unnamed: 0'])

# Verify the changes
print("Engage DataFrame after dropping 'Unnamed: 0' column:")
print(engage_df.head())


# In[5]:


# Find duplicate student IDs
duplicate_students = engage_df[engage_df.duplicated('STUDENT_ID_NUMBER', keep=False)]

# Print the rows with duplicate student IDs
print("Rows with duplicate student IDs:")
print(duplicate_students)


# In[6]:


import pandas as pd

# Drop the 'EngagementClass' column and rows with 'EventParticipationCount' = 1
engage_df = engage_df.drop(columns=['EngagementClass'])
engage_df = engage_df[engage_df['EventParticipationCount'] > 1]

# Calculate the quartiles
quartiles = engage_df['EventParticipationCount'].quantile([0.25, 0.5, 0.75])

# Define the classification function based on quartiles
def classify_event_participation(count):
    if count <= quartiles[0.25]:
        return 'Low'
    elif count <= quartiles[0.75]:
        return 'Medium'
    else:
        return 'High'

# Apply the classification
engage_df['EngagementClass'] = engage_df['EventParticipationCount'].apply(classify_event_participation)

# Keep only the specified features
final_engage_df = engage_df[['STUDENT_ID_NUMBER', 'GENDER_CODE', 'DATE_OF_BIRTH', 'RESIDENCY_DESC', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC', 'EventParticipationCount', 'EngagementClass']]

# Convert birth year to age
current_year = pd.to_datetime('today').year
final_engage_df['Age'] = current_year - final_engage_df['DATE_OF_BIRTH']

# Categorize residency as Home or International
def classify_residency(residency):
    if 'Home' in residency:
        return 'Home'
    else:
        return 'International'

final_engage_df['Residency'] = final_engage_df['RESIDENCY_DESC'].apply(classify_residency)

# Drop the original 'DATE_OF_BIRTH' and 'RESIDENCY_DESC' columns
final_engage_df = final_engage_df.drop(columns=['DATE_OF_BIRTH', 'RESIDENCY_DESC'])

# Define the mapping function for YEAR_CODE
def map_year_code(year_code):
    if 'A' in year_code:
        return 0  # Convert '2A' to 0, '3A' to 0, etc.
    elif len(year_code) == 2 and year_code[0] == '0':
        return int(year_code[1])  # Convert '01' to 1, '02' to 2, etc.
    else:
        return int(year_code)  # Convert numeric strings directly to integers

# Apply the mapping function to the YEAR_CODE column
final_engage_df['YEAR_CODE'] = final_engage_df['YEAR_CODE'].apply(map_year_code)

# Verify the final DataFrame
print(final_engage_df.head())


# In[7]:


# Check the number of rows in the final dataframe
num_rows = final_engage_df.shape[0]
print(f"Number of rows in final_engage_df: {num_rows}")

# Calculate and print the quantile values for EventParticipationCount
quartiles = final_engage_df['EventParticipationCount'].quantile([0.25, 0.5, 0.75])
print("Quantile values for EventParticipationCount:")
print(quartiles)

# Calculate the number of students in each engagement class
engagement_class_counts = final_engage_df['EngagementClass'].value_counts()
print("Number of students in each engagement class:")
print(engagement_class_counts)


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to plot stacked 100% column chart
def plot_stacked_bar(df, feature, title):
    crosstab = pd.crosstab(df[feature], df['EngagementClass'], normalize='index') * 100
    crosstab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Percentage')
    plt.legend(loc='upper right', title='Engagement Class')
    plt.show()

# Plot stacked 100% column charts for each feature
features = ['GENDER_CODE', 'Age', 'Residency', 'YEAR_CODE', 'FT_PT_IND', 'COLL_DESC']
for feature in features:
    plot_stacked_bar(final_engage_df, feature, f'Distribution of Engagement Classes by {feature}')


# In[9]:


# Function to generate and print crosstab for a feature
def print_crosstab(df, feature):
    crosstab = pd.crosstab(df[feature], df['EngagementClass'], normalize='index') * 100
    print(f"Crosstab for {feature}:")
    print(crosstab)
    print("\n")

# Generate and print crosstabs for each feature
for feature in features:
    print_crosstab(final_engage_df, feature)


# In[10]:


# Check the number of students in each category for significant features

# Number of students by GENDER_CODE
gender_counts = final_engage_df['GENDER_CODE'].value_counts()
print("Number of students by GENDER_CODE:")
print(gender_counts)

# Number of students by Age
age_counts = final_engage_df['Age'].value_counts()
print("\nNumber of students by Age:")
print(age_counts)

# Number of students by Residency
residency_counts = final_engage_df['Residency'].value_counts()
print("\nNumber of students by Residency:")
print(residency_counts)

# Number of students by YEAR_CODE
year_code_counts = final_engage_df['YEAR_CODE'].value_counts()
print("\nNumber of students by YEAR_CODE:")
print(year_code_counts)

# Number of students by FT_PT_IND
ft_pt_ind_counts = final_engage_df['FT_PT_IND'].value_counts()
print("\nNumber of students by FT_PT_IND:")
print(ft_pt_ind_counts)

# Number of students by COLL_DESC
coll_desc_counts = final_engage_df['COLL_DESC'].value_counts()
print("\nNumber of students by COLL_DESC:")
print(coll_desc_counts)

# Number of students in each Engagement Class
engagement_class_counts = final_engage_df['EngagementClass'].value_counts()
print("\nNumber of students in each Engagement Class:")
print(engagement_class_counts)


# ### drop full time and part time columns, very little data in part time to say anything conclusive
# 
# ### drop outliers from age (28+) and year code (0,6,7)
# 
# ### non binary gender also only has 19 values
# 
# ### inter faculty has 35 
# 
# 
# 

# ### Biological science has the highest percentage of high, but only 310 students, whereas social science w second highest number of students has the second highest high % at 26 followed by env (557 students) at 25 
# 
# 
# ### Engagement from year 1 to 4 decreases (3 higher than 4, but 4 has half the number of students as 3), this implies that trend of decreasing engagement is strongest from 1 to 3
# 
# ### Age has a similar decrease as year but is not as strong 
# 
# 

# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions before refining
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(final_engage_df['Age'], bins=20, kde=True)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(final_engage_df['YEAR_CODE'], bins=range(1, 6), kde=False, discrete=True)
plt.title('Year Code Distribution')


plt.tight_layout()
plt.show()

# Checking other categorical variables distributions before refining
print(final_engage_df['GENDER_CODE'].value_counts())
print(final_engage_df['Residency'].value_counts())
print(final_engage_df['COLL_DESC'].value_counts())
print(final_engage_df['FT_PT_IND'].value_counts())


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for the first part of the first chart
plt.figure(figsize=(12, 6))

# Plot Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(final_engage_df['Age'], bins=20, kde=True)
plt.title('Age Distribution')

# Plot Year Code Distribution
plt.subplot(1, 2, 2)
sns.histplot(final_engage_df['YEAR_CODE'], bins=range(1, 6), kde=False, discrete=True)
plt.title('Year Code Distribution')

plt.tight_layout()
plt.show()


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for the second part of the first chart
plt.figure(figsize=(12, 6))

# Plot Gender Code Distribution
plt.subplot(1, 2, 1)
sns.countplot(x='GENDER_CODE', data=final_engage_df)
plt.title('Gender Code Distribution')

# Plot Residency Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='Residency', data=final_engage_df)
plt.title('Residency Distribution')

plt.tight_layout()
plt.show()


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for the second chart
plt.figure(figsize=(12, 6))

# Plot College Description Distribution with rotated x labels
plt.subplot(1, 2, 1)
sns.countplot(x='COLL_DESC', data=final_engage_df)
plt.title('College Description Distribution')
plt.xticks(rotation=45, ha='right')

# Plot Full-Time/Part-Time Indicator Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='FT_PT_IND', data=final_engage_df)
plt.title('Full-Time/Part-Time Indicator Distribution')

plt.tight_layout()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Creating a refined dataset
refined_engage_df = final_engage_df.copy()

# Dropping part-time students due to low sample size
refined_engage_df = refined_engage_df[refined_engage_df['FT_PT_IND'] == 'FULLTIME']

# Dropping age outliers (students older than 28)
refined_engage_df = refined_engage_df[(refined_engage_df['Age'] >= 19) & (refined_engage_df['Age'] <= 28)]

# Dropping year code outliers (year codes 0, 6, and 7)
refined_engage_df = refined_engage_df[~refined_engage_df['YEAR_CODE'].isin([0, 6, 7])]

# Dropping non-binary gender due to low sample size
refined_engage_df = refined_engage_df[refined_engage_df['GENDER_CODE'] != 'N']

# Dropping inter-faculty and small groups
refined_engage_df = refined_engage_df[~refined_engage_df['COLL_DESC'].isin(['Inter-faculty'])]

# Dropping columns that will not be used for modeling
refined_engage_df = refined_engage_df.drop(columns=['FT_PT_IND'])

# Recalculate EngagementClass
quartiles = refined_engage_df['EventParticipationCount'].quantile([0.25, 0.5, 0.75])

def classify_engagement(participation_count):
    if participation_count <= quartiles[0.25]:
        return 'Low'
    elif participation_count <= quartiles[0.75]:
        return 'Medium'
    else:
        return 'High'

refined_engage_df['EngagementClass'] = refined_engage_df['EventParticipationCount'].apply(classify_engagement)

# Plot distributions after refining
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(refined_engage_df['Age'], bins=20, kde=True)
plt.title('Age Distribution After Refining')

plt.subplot(1, 2, 2)
sns.histplot(refined_engage_df['YEAR_CODE'], bins=range(1, 6), kde=False, discrete=True)
plt.title('Year Code Distribution After Refining')

plt.tight_layout()
plt.show()

# Verify the refined dataframe
print("Refined Engage DataFrame:")
print(refined_engage_df.head())


# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for the second part of the first chart
plt.figure(figsize=(12, 6))

# Plot Gender Code Distribution
plt.subplot(1, 2, 1)
sns.countplot(x='GENDER_CODE', data=refined_engage_df)
plt.title('Gender Code Distribution After Refining')

# Plot Residency Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='COLL_DESC', data=refined_engage_df)
plt.title('College Description Distribution After Refining')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[13]:


# Plot distributions after refining
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(refined_engage_df['Age'], bins=10, kde=True)
plt.title('Age Distribution After Refining')

plt.subplot(1, 2, 2)
sns.histplot(refined_engage_df['YEAR_CODE'], bins=range(1, 6), kde=False, discrete=True)
plt.title('Year Code Distribution After Refining')

plt.tight_layout()
plt.show()

# Checking other categorical variables distributions after refining
print(refined_engage_df['GENDER_CODE'].value_counts())
print(refined_engage_df['Residency'].value_counts())
print(refined_engage_df['COLL_DESC'].value_counts())


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to plot stacked 100% column chart
def plot_stacked_bar(ax, df, feature, title):
    crosstab = pd.crosstab(df[feature], df['EngagementClass'], normalize='index') * 100
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel('Percentage')
    ax.legend(loc='upper right', title='Engagement Class')

# Number of features
features = ['GENDER_CODE', 'Age', 'Residency', 'YEAR_CODE', 'COLL_DESC']
n_features = len(features)

# Determine grid size
n_cols = 2  # Number of columns
n_rows = (n_features + 1) // n_cols  # Number of rows (ensure all features fit in the grid)

# Create subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
axs = axs.flatten()  # Flatten to make indexing easier

# Plot each feature in a grid
for i, feature in enumerate(features):
    plot_stacked_bar(axs[i], refined_engage_df, feature, f'Distribution of Engagement Classes by {feature}')

# Remove any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


# In[53]:


# Function to generate and print crosstab for a feature
def print_crosstab(df, feature):
    crosstab = pd.crosstab(df[feature], df['EngagementClass'], normalize='index') * 100
    print(f"Crosstab for {feature}:")
    print(crosstab)
    print("\n")

# Generate and print crosstabs for each feature
for feature in features:
    print_crosstab(refined_engage_df, feature)


# # Random Forest

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Drop unnecessary columns
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EventParticipationCount', 'EngagementClass'])
y = refined_engage_df['EngagementClass']

# Encode categorical variables
X = pd.get_dummies(X, columns=['GENDER_CODE', 'Residency', 'COLL_DESC'], drop_first=True)

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[16]:


# Get feature importances
feature_importances = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances in Random Forest Model')
plt.show()


# In[17]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create the RandomizedSearchCV object
rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the RandomizedSearchCV object to the data
rf_random.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", rf_random.best_params_)

# Evaluate the model with the best parameters
best_rf_model = rf_random.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

print("Random Forest Classification Report with Best Parameters")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("Accuracy:", accuracy_score(y_test, y_pred_best))


# # XGBoost

# In[18]:


import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Prepare the data for XGBoost
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount'])
y = refined_engage_df['EngagementClass']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['GENDER_CODE', 'Residency','COLL_DESC'])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Calculate scale_pos_weight for class balancing
class_counts = pd.Series(y_train).value_counts()
scale_pos_weight = class_counts.min() / class_counts

# Train the XGBoost model with class balancing
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                              scale_pos_weight=scale_pos_weight.to_dict())
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("XGBoost Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[19]:


import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Prepare the data for XGBoost
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount','GENDER_CODE', 'Residency','COLL_DESC','YEAR_CODE'])
y = refined_engage_df['EngagementClass']


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Calculate scale_pos_weight for class balancing
class_counts = pd.Series(y_train).value_counts()
scale_pos_weight = class_counts.min() / class_counts

# Train the XGBoost model with class balancing
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                              scale_pos_weight=scale_pos_weight.to_dict())
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("XGBoost Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[37]:


pip install lazypredict


# In[20]:


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Prepare the data for modeling
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount'])
y = refined_engage_df['EngagementClass']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the categorical columns
categorical_cols = ['GENDER_CODE', 'Residency', 'COLL_DESC']

# Create a column transformer with OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))  # Handle any missing values if present
])

# Transform the data
X_processed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Use LazyClassifier to train and evaluate multiple models
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Print the results
print(models)


# In[21]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Prepare the data for modeling
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount'])
y = refined_engage_df['EngagementClass']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the categorical columns
categorical_cols = ['GENDER_CODE', 'Residency', 'COLL_DESC']

# Create a column transformer with OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))  # Handle any missing values if present
])

# Transform the data
X_processed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("SVM Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[43]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Prepare the data for modeling
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount'])
y = refined_engage_df['EngagementClass']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the categorical columns
categorical_cols = ['GENDER_CODE', 'Residency', 'COLL_DESC']

# Create a column transformer with OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and SimpleImputer
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# Transform the data
X_processed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'class_weight': ['balanced', None]
}

# Initialize the SVM model
svm_model = SVC()

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict on the test set with the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Best Parameters:", best_params)
print("SVM Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# ## Two class df

# In[22]:


import pandas as pd

# Create a copy of the refined DataFrame
two_class_df = refined_engage_df.copy()

# Calculate the quantiles for EventParticipationCount
quartiles = two_class_df['EventParticipationCount'].quantile([0.75])

# Define the classification function based on the 0.75 quantile
def classify_two_class(participation_count):
    if participation_count > quartiles[0.75]:
        return 'High'
    else:
        return 'Medium'

# Apply the classification
two_class_df['EngagementClass'] = two_class_df['EventParticipationCount'].apply(classify_two_class)

# Verify the changes
print(two_class_df['EngagementClass'].value_counts())
print(two_class_df.head())


# In[23]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Prepare the data for modeling
X = two_class_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount'])
y = two_class_df['EngagementClass']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the categorical columns
categorical_cols = ['GENDER_CODE', 'Residency', 'COLL_DESC']

# Create a column transformer with OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and SimpleImputer
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# Transform the data
X_processed = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize the Random Forest model with class balancing
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[25]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Prepare the data
X = two_class_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount','COLL_DESC'])
y = two_class_df['EngagementClass']

# Encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Age', 'YEAR_CODE']),
        ('cat', OneHotEncoder(), ['GENDER_CODE', 'Residency'])
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessor.fit_transform(X), y, test_size=0.2, random_state=42)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# In[50]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define the model
rf_model = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train_encoded)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classification Report")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))

print("Accuracy:", accuracy_score(y_test_encoded, y_pred))


# ## SMOTE

# In[33]:


refined_engage_df.to_csv('/Users/pranshu/Desktop/DissData/refined_engage.csv')


# In[34]:


two_class_df.to_csv('/Users/pranshu/Desktop/DissData/two_class.csv')


# In[31]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Assuming 'refined_engage_df' is already defined and contains the necessary data.

# Prepare the data
X = refined_engage_df.drop(columns=['STUDENT_ID_NUMBER', 'EngagementClass', 'EventParticipationCount'])
y = refined_engage_df['EngagementClass']

# One-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Age', 'YEAR_CODE']),
        ('cat', OneHotEncoder(), ['GENDER_CODE', 'Residency', 'COLL_DESC'])
    ])

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Transform features
X_transformed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Check the distribution of the new training set
print("Distribution of classes in y_train_res:")
print(pd.Series(y_train_res).value_counts())

# Define the RandomForest model
rf_model = RandomForestClassifier(random_state=42)

# Perform hyperparameter tuning with grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("Tuned Random Forest Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:




