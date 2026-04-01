from sklearn.preprocessing import StandardScaler

def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_test_scaler = scaler.transform(X_test)
    X_train_scaler = scaler.fit_transform(X_train)
    return X_test_scaler,X_train_scaler

