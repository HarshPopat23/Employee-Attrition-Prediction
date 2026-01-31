from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic(X_train_scaled, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
