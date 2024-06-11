import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('C:/Users/User/Desktop/medicalDiagnosis/datasets/dataset.csv')
df = df.replace('_', ' ', regex=True)
df = df.fillna(0)

# Create symptom severity DataFrame
df1 = pd.read_csv('C:/Users/User/Desktop/medicalDiagnosis/datasets/Symptom-severity.csv')
x = df1['Symptom']
dfx = pd.DataFrame({'Disease': df['Disease']})
dfx[x] = 0

# Populate symptom severity DataFrame
for index, row in df.iterrows():
    for symptom in df.columns[1:]:
        if row[symptom] != 0:
            dfx.loc[index, row[symptom]] = 1


# # Clean and preprocess data
dfx = dfx.drop(columns=['foul_smell_ofurine', 'dischromic_patches', 'spotting_urination'])
# dfx[dfx.columns[1:]] = dfx[dfx.columns[1:]].astype(int)
dfx[dfx.columns[1:]].sum(axis=0).sort_values()

# Prepare data and labels
data = dfx.iloc[:, 1:].values
labels = dfx['Disease'].values
y=df['Disease'].unique()
# Split data into train, test, and validation sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.3, random_state=42)

#Preprocessing

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val=le.transform(y_val)
y=le.classes_


# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True),
    'GradientBoost': GradientBoostingClassifier(),
    'ExtraTrees': ExtraTreesClassifier()
}

# Train and evaluate classifiers
for name, clf in classifiers.items():
    # Cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_scores = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='f1_weighted')
    print(f'{name} cross-validation mean F1 score: %.3f' % cv_scores.mean())

    # Train
    clf.fit(x_train, y_train)
    
    # Test
    test_predictions = clf.predict(x_test)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    test_roc = roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovr')
    print(f'{name} test F1 Score: {test_f1:.4f}, AUC-ROC Score: {test_roc:.4f}')
    
    # Validation
    val_predictions = clf.predict(x_val)
    val_f1 = f1_score(y_val, val_predictions, average='weighted')
    val_roc = roc_auc_score(y_val, clf.predict_proba(x_val), multi_class='ovr')
    print(f'{name} validation F1 Score: {val_f1:.4f}, AUC-ROC Score: {val_roc:.4f}')

    # Save model
    pickle.dump(clf, open(f"{name}.pkl", "wb"))

# Load description and precaution data
desc = pd.read_csv("C:/Users/User/Desktop/medicalDiagnosis/datasets/symptom_Description.csv")
prec = pd.read_csv("C:/Users/User/Desktop/medicalDiagnosis/datasets/symptom_precaution.csv")

# Define prediction function
def predict_disease(model, symptoms):
    # Load model
    with open(model, 'rb') as f:
        clf = pickle.load(f)
    
    # Prepare input data
    t = pd.Series(0, index=dfx.columns[1:])
    t[symptoms] = 1
    t = t.to_numpy().reshape(1, -1)
    
    # Make prediction
    proba = clf.predict_proba(t)
    
    # Get top 5 predicted diseases
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]
    top5_diseases = labels[top5_idx]

    # Print predictions
    for i in range(5):
        disease = top5_diseases[i]
        probability = top5_proba[i]
        
        print("Disease Name: ", disease)
        print("Probability: ", probability)
        
        if disease in desc["Disease"].unique():
            disp = desc[desc['Disease'] == disease]['Description'].values[0]
            print("Disease Description: ", disp)
        
        if disease in prec["Disease"].unique():
            precuations = prec[prec['Disease'] == disease].iloc[:, 1:].values[0]
            print("Recommended Precautions: ")
            for precaution in precuations:
                print(precaution)
        
        print("\n")
# for name, clf in classifiers.items():
#     # Train the classifier
#     clf.fit(x_train, y_train)
    
#     # Save the trained classifier to a file
#     with open(f"{name}.pkl", "wb") as f:
#         pickle.dump(clf, f)

# best_classifier = None
# best_score = -1

# for name, clf in classifiers.items():
#     # Cross-validation
#     kfold = KFold(n_splits=10, shuffle=True, random_state=1)
#     cv_scores = cross_val_score(clf, x_train, y_train, cv=kfold, scoring='f1_weighted')
#     mean_cv_score = cv_scores.mean()
    
#     # Check if this classifier has the best score
#     if mean_cv_score > best_score:
#         best_score = mean_cv_score
#         best_classifier = clf

# # Train the best classifier on the full training data
# best_classifier.fit(x_train, y_train)

# # Save the best classifier to a file
# with open("best_classifier.pkl", "wb") as f:
#     pickle.dump(best_classifier, f)

# # Example usage
# symptoms = ['chest pain', 'phlegm', 'runny nose', 'high fever', 'throat irritation', 'congestion', 'redness of eyes']
# predict_disease("ExtraTrees.pkl", symptoms)





