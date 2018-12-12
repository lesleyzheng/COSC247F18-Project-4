import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class modelSelection(object):

    def __init__(self):

        self.master_lat = None
        self.master_long = None
        self.master_loc = None

        self.raw_master_features = None
        self.master_features = None
        self.master_test_features = None
        self.test_dict = None

        self.scaler = None


    def start(self):

        self.initializeValues()
        # Decision Tree
        # print("Regression Decision Tree")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_dt()

        # SVR
        # print("SVR")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_svr()

        # SVR linear
        # print("SVR")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_svr_linear()

        # linear regression
        print("linear regression_v1")
        x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_svr_linear()

        # error
        total_error = self.total_MSE(x_train_predicts, self.master_lat, y_train_predicts, self.master_long)

        # print("norm")
        # x_predicts, y_predicts = self.SVM(False)
        # self.total_MSE(x_predicts, self.master_lat, y_predicts, self.master_long)

        # print("advanced")
        # x_predicts, y_predicts = self.kNN_advanced()
        # self.total_MSE(x_predicts, self.master_lat, y_predicts, self.master_long)

        new_file = open("./data/submission_linear_regression_v1.txt", "w")


        counter = 0
        new_file.write("Id,Lat,Lon")
        for key in self.test_dict.keys():
            string = "\n" + str(key) + "," + str(x_test_predicts[counter]) + "," + str(y_test_predicts[counter])
            new_file.write(string)
            counter += 1
        new_file.close()

    def total_MSE(self, pred_x, targ_x, pred_y, targ_y):

        location = pred_x + pred_y
        pred_location = targ_x + targ_y

        loss = (mean_squared_error(location, pred_location)) ** (1 / 2)

        print("Total MSE " + str(loss))

    def linear_regression(self):

        learner = LinearRegression()
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        learner.fit(self.master_features, self.master_long)

        long_train_preds = learner.predict(self.master_features)
        long_test_preds = learner.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def kNN_norm(self, test):

        # kNN
        kNN = KNeighborsRegressor(n_neighbors=10, n_jobs= 2)

        kNN.fit(self.master_features, self.master_lat)
        if test:
            lat_predicts = kNN.predict(self.master_test_features)
        else:
            lat_predicts = kNN.predict(self.master_features)

        kNN.fit(self.master_features, self.master_long)
        if test:
            long_predicts = kNN.predict(self.master_test_features)
        else:
            long_predicts = kNN.predict(self.master_features)

        return lat_predicts, long_predicts

    def kNN_advanced(self):

        #kNN
        kNN = KNeighborsRegressor(n_neighbors=1, n_jobs= 2)

        kNN.fit(self.master_features, self.master_lat)
        lat_predicts = kNN.predict(self.master_features)
        n, m = self.raw_master_features.shape
        print(str(n) + " " + str(m))
        new_master_features = np.hstack((self.master_features, lat_predicts.reshape((n, 1))))
        n1, m1 = new_master_features.shape
        print(str(n1) + " " + str(m1))

        new_scaler = MinMaxScaler()
        new_scaler.fit(new_master_features)
        new_master_features = new_scaler.transform(new_master_features)

        kNN.fit(new_master_features, self.master_long)
        long_predicts = kNN.predict(new_master_features)

        return lat_predicts, long_predicts

    def SVM(self, test):

        SVM = SVR(kernel = "rbf", max_iter = 10000)

        SVM.fit(self.master_features, self.master_lat)
        if test:
            lat_predicts = SVM.predict(self.master_test_features)
        else:
            lat_predicts = SVM.predict(self.master_features)

        SVM.fit(self.master_features, self.master_long)
        if test:
            long_predicts = SVM.predict(self.master_test_features)
        else:
            long_predicts = SVM.predict(self.master_features)

        return lat_predicts, long_predicts

    def runGridSearch_dt(self):

        params = [{'splitter': ['best', 'random'], 'max_depth': [None, 1, 2, 3, 4, 5, 6]}]

        learner = DecisionTreeRegressor()
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs=15)
        gs.fit(self.master_features, self.master_lat)

        print("The best parameters for latitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_train_preds = gs.predict(self.master_features)
        lat_test_preds = gs.predict(self.master_test_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_train_preds = gs.predict(self.master_features)
        long_test_preds = gs.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def runGridSearch_svr_linear(self):

        print("linear")

        params = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        learner = SVR()
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)

        print("The best parameters for latitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_train_preds = gs.predict(self.master_features)
        lat_test_preds = gs.predict(self.master_test_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_train_preds = gs.predict(self.master_features)
        long_test_preds = gs.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def runGridSearch_svr_rbf(self):

        params = [
            {'kernel': ['rbf'], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]},
                 ]

        learner = SVR()
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)

        print("The best parameters for latitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_train_preds = gs.predict(self.master_features)
        lat_test_preds = gs.predict(self.master_test_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_train_preds = gs.predict(self.master_features)
        long_test_preds = gs.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def runGridSearch_svr_poly(self):

        params = [{'kernel': ['poly'], 'coef0': [.1, 1, 10, 100], 'degree': [2, 3, 4, 5], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]},
                  ]

        learner = SVR()
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)

        print("The best parameters for latitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_train_preds = gs.predict(self.master_features)
        lat_test_preds = gs.predict(self.master_test_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_train_preds = gs.predict(self.master_features)
        long_test_preds = gs.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def runGridSearch_svr_sigmoid(self):

        params = [{'kernel': ['sigmoid'], 'coef0': [.1, 1, 10, 100], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]}]

        learner = SVR()
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)

        print("The best parameters for latitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_train_preds = gs.predict(self.master_features)
        lat_test_preds = gs.predict(self.master_test_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_train_preds = gs.predict(self.master_features)
        long_test_preds = gs.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def runGridSearch_svr(self):

        params = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
            {'kernel': ['rbf'], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]},
                  {'kernel': ['poly'], 'coef0': [.1, 1, 10, 100], 'degree': [2, 3, 4, 5], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]},
                  {'kernel': ['sigmoid'], 'coef0': [.1, 1, 10, 100], 'gamma': [1.0, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]}]

        learner = SVR()
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)

        print("The best parameters for latitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_train_preds = gs.predict(self.master_features)
        lat_test_preds = gs.predict(self.master_test_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_train_preds = gs.predict(self.master_features)
        long_test_preds = gs.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def runGridSearch_knn(self):

        params = {'n_neighbors': [1,3,5,7,9], 'weights' : ['uniform', 'distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'], 'p' : [1,2]}

        learner = KNeighborsRegressor(n_jobs = 15)
        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)
        print("The best parameters found were: ")
        print(gs.best_params_)
        print(gs.get_params())


        lat_test_preds = gs.predict(self.master_test_features)

        lat_train_preds = gs.predict(self.master_features)

        gs.fit(self.master_features, self.master_long)
        print("the best params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())
        long_test_preds = gs.predict(self.master_test_features)

        long_train_preds = gs.predict(self.master_features)
        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

        #Decision Tree
    def decisionTree(self, test):
        dTree = DecisionTreeRegressor(max_depth = 5)

        dTree.fit(self.master_features, self.master_lat)
        if test:
            lat_predicts = dTree.predict(self.master_test_features)
        else:
            lat_predicts = dTree.predict(self.master_features)

        dTree.fit(self.master_features, self.master_long)
        if test:
            long_predicts = dTree.predict(self.master_test_features)
        else:
            long_predicts = dTree.predict(self.master_features)

        return lat_predicts, long_predicts


    def initializeValues(self):

        # load files
        pickle_in = open("./data/target_values.pkl", "rb")
        (self.master_lat, self.master_long, self.master_loc), desc = pickle.load(pickle_in)
        print(desc)

        pickle_in2 = open("./data/master_features.pkl", "rb")
        self.master_features, desc2 = pickle.load(pickle_in2)
        print(desc2)

        pickle_in3 = open("./data/master_test_features.pkl", "rb")
        self.master_test_features, desc3 = pickle.load(pickle_in3)
        print(desc3)

        pickle_in4 = open("./data/posts_test_dict.pkl", "rb")
        self.test_dict, desc4 = pickle.load(pickle_in4)
        print(desc4)

        # convert to np array
        self.master_features = np.array(self.master_features)
        self.master_test_features = np.array(self.master_test_features)
        self.master_lat = self.get_target_array(self.master_lat)
        self.master_long = self.get_target_array(self.master_long)
        self.master_loc = self.get_target_array(self.master_loc)

        # scaleValues
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.master_features)
        self.raw_master_features = self.master_features
        self.master_features = self.scaler.transform(self.master_features)
        self.master_test_features = self.scaler.transform(self.master_test_features)

    def get_target_array(self, target_dict):

        target_list = list(target_dict.values())
        return np.array(target_list)


if __name__ == "__main__":

    myModelSelection = modelSelection()
    myModelSelection.start()