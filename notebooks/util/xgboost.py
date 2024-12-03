from xgboost import XGBClassifier
from imblearn.over_sampling import KMeansSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from util.get_data import get_features_and_targets
from util.model_eval import predict_and_score

# Building the most successful xgboost model in one place, as learned from the xgboost notebook
def best_xgb_model():
    # Get features and targets
    features, targets = get_features_and_targets()

    # Apply scaling and K-means SMOTE to balance the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled =  pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

    features_train, features_test, targets_train, targets_test = train_test_split(features_scaled, targets, test_size=0.3, random_state=0)

    kmeans_smote = KMeansSMOTE(
        random_state=0,
        k_neighbors=5,
        cluster_balance_threshold=0.1,
        n_jobs=-1
    )

    features_train_resampled, targets_train_resampled = kmeans_smote.fit_resample(features_train, targets_train)

    # The best xgboost model used no demographic data, and only the last 3 months of payment data
    features_train_no_demographics = features_train_resampled.drop(['GENDER', 'AGE', 'EDUCATION_LEVEL', 'MARITAL_STATUS'], axis=1)
    features_test_no_demographics = features_test.drop(['GENDER', 'AGE', 'EDUCATION_LEVEL', 'MARITAL_STATUS'], axis=1)
    features_train_no_demo_most_recent = features_train_no_demographics.drop(['APRIL_BILL', 'APRIL_PAYMENT', 'MAY_BILL', 'MAY_PAYMENT', 'JUNE_BILL', 'JUNE_PAYMENT'], axis=1)
    features_test_no_demo_most_recent = features_test_no_demographics.drop(['APRIL_BILL', 'APRIL_PAYMENT', 'MAY_BILL', 'MAY_PAYMENT', 'JUNE_BILL', 'JUNE_PAYMENT'], axis=1)

    # Fit the model with most optimal parameters found
    xgb_model_no_demo_most_recent = XGBClassifier(eval_metric='logloss', learning_rate=0.05, max_depth=3, n_estimators=100)
    xgb_model_no_demo_most_recent.fit(features_train_no_demo_most_recent, targets_train_resampled)

    # Evaluate the model
    predict_and_score(xgb_model_no_demo_most_recent, features_test_no_demo_most_recent, targets_test)

    return xgb_model_no_demo_most_recent