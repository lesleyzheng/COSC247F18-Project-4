import numpy as np
import  pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def run_log_reg(X_tr, y_tr):
    learner = LogisticRegression()
    learner.fit(X_tr, y_tr)
    train_preds = learner.predict(X_tr)
    train_loss = zero_one_loss(y_tr, train_preds)
    train_loss = round(train_loss, 4)

    print(f"The train loss is {train_loss}")


def run_KNN(X_tr, y_tr):
    learner = KNeighborsClassifier(n_neighbors=3)
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    new_X_tr = scaler.transform(X_tr)

    learner.fit(new_X_tr, y_tr)

    train_preds = learner.predict(new_X_tr)
    train_loss = zero_one_loss(y_tr, train_preds)
    train_loss = round(train_loss, 4)

    print(f"The train loss is {train_loss}")


def run_grid_search(X_tr, y_tr):
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    new_X_tr = scaler.transform(X_tr)

    params = [{'kernel': ['rbf'], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'C': [1, 10, 100, 1000]},
              {'kernel': ['sigmoid'], 'coef0': [.1, 1, 10, 100], 'C': [1, 10, 100, 1000]}]

    learner = SVC()
    gs = GridSearchCV(learner, params, 'f1', cv=5)
    gs.fit(new_X_tr, y_tr)

    print("The best parameters found were: ")
    print(gs.best_params_)