import numpy as np
import pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def run_lin_reg(X_tr, y_tr, X_te):
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    new_X_tr = scaler.transform(X_tr)
    learner = LinearRegression()
    learner.fit(new_X_tr, y_tr)
    train_preds = learner.predict(new_X_tr)
    train_loss = mean_squared_error(y_tr, train_preds)
    # train_loss = round(train_loss, 4)

    print(f"The train loss is {train_loss}")
    new_X_te = scaler.transform(X_te)
    test_preds = learner.predict(new_X_te)
    return test_preds


def run_KNN(X_tr, y_tr, X_te):
    learner = KNeighborsRegressor(n_neighbors=3)
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    new_X_tr = scaler.transform(X_tr)

    learner.fit(new_X_tr, y_tr)

    train_preds = learner.predict(new_X_tr)
    train_loss = mean_squared_error(y_tr, train_preds)

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


def get_target_array(target_dict):
    target_list = list(target_dict.values())

    return np.array(target_list)


if __name__ == '__main__':
    pickle_in = open("./data/target_values.pkl", "rb")
    (master_lat, master_long, master_loc), desc = pickle.load(pickle_in)

    pickle_in2 = open("./data/master_features.pkl", "rb")
    master_features, desc2 = pickle.load(pickle_in2)

    pickle_in3 = open("./data/master_test_features.pkl", "rb")
    master_test_features, desc3 = pickle.load(pickle_in3)

    master_test_features = np.array(master_test_features)
    master_features = np.array(master_features)
    target_array = get_target_array(master_lat)

    test_preds_lat = run_lin_reg(master_features, target_array, master_test_features)

    target_array = get_target_array(master_long)
    test_preds_long = run_lin_reg(master_features, target_array, master_test_features)

    pickle_in4 = open("./data/posts_test_dict.pkl", "rb")
    test_dict, desc4 = pickle.load(pickle_in4)

    new_file = open("./data/submission_linreg.txt", "w")

    counter = 0
    new_file.write("Id,Lat,Lon")
    for key in test_dict.keys():
        string = str(key) + "," + str(test_preds_lat[counter]) + "," + str(test_preds_long[counter]) + "/n"
        new_file.write(string)
        counter +=1
    new_file.close()
