from sklearn.metrics import classification_report, roc_auc_score

def predict_and_score(model, features_test, targets_test):
    pred = model.predict(features_test)
    clf_report = classification_report(targets_test, pred)
    roc_auc = roc_auc_score(targets_test, model.predict_proba(features_test)[:, 1])
    print("\nClassification Report:")
    print(clf_report)
    print('ROC-AUC Score: {}'.format(roc_auc))
    return clf_report, roc_auc, pred 