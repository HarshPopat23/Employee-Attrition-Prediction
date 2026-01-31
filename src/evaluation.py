from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test, scaled=False):
    if scaled:
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
    else:
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
