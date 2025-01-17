import pandas as pd
import numpy as np

# File paths
diabetic_data_path = './dataset/diabetic_data.csv'
ids_mapping_path = './dataset/IDs_mapping.csv'

# Load datasets
diabetic_data = pd.read_csv(diabetic_data_path)
ids_mapping = pd.read_csv(ids_mapping_path)

# Step 1: Handle missing values
# Replace '?' with NaN
diabetic_data.replace('?', np.nan, inplace=True)

# Drop columns with more than 50% missing data
threshold = 0.5 * diabetic_data.shape[0]
diabetic_data.dropna(axis=1, thresh=threshold, inplace=True)

# Fill remaining missing values with mode (categorical) or median (numerical)
for col in diabetic_data.columns:
    if diabetic_data[col].dtype == 'object':
        diabetic_data[col].fillna(diabetic_data[col].mode()[0], inplace=True)
    else:
        diabetic_data[col].fillna(diabetic_data[col].median(), inplace=True)

# Step 2: Map categorical IDs to descriptions
# Example: Map 'admission_type_id' to its description from IDs_mapping
admission_mapping = ids_mapping[ids_mapping['admission_type_id'].notnull()]
admission_mapping_dict = admission_mapping.set_index('admission_type_id')['description'].to_dict()

diabetic_data['admission_type_id'] = diabetic_data['admission_type_id'].map(admission_mapping_dict)

# Step 3: Encode categorical features
# Convert target variable 'readmitted' into binary
diabetic_data['readmitted'] = diabetic_data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# One-hot encode categorical columns
categorical_cols = diabetic_data.select_dtypes(include=['object']).columns
diabetic_data = pd.get_dummies(diabetic_data, columns=categorical_cols, drop_first=True)

# Step 4: Feature engineering
# Remove irrelevant columns (e.g., 'encounter_id', 'patient_nbr')
diabetic_data.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

# Step 5: Save cleaned dataset
cleaned_data_path = './dataset/cleaned_diabetic_data.csv'
diabetic_data.to_csv(cleaned_data_path, index=False)

print(f"Data preprocessing complete. Cleaned dataset saved at {cleaned_data_path}.")
