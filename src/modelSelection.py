import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score


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

        # v2
        self.raw_master_features_v2 = None
        self.master_features_v2 = None
        self.master_test_features_v2 = None
        self.scaler_v2 = None

    def start(self):

        self.initializeValues()

        # Decision Tree
        # print("Regression Decision Tree")
        # x_test_predicts, y_test_predicts = self.decisionTree(True)

        # Random Forest Grid Search
        # print("Random Forest Grid Search")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_random_forest()

        # Decision Tree Grid Search
        # print("Regression Decision Tree GS")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_dt()


        #Bagging
        # print("Bagging using decision tree")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.bagging()

        # SVR Grid Search
        # print("SVR GS")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_svr()


        # error
        # total_svr_error = self.total_MSE(x_train_predicts, self.master_lat, y_train_predicts, self.master_long)


        # print("norm")
        # x_test_predicts, y_test_predicts = self.kNN_norm(True)
        # self.total_MSE(x_train_predicts, self.master_lat, y_train_predicts, self.master_long)

        # SVR linear
        # print("SVR GS linear")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.runGridSearch_svr_linear()

        # linear regression
        # print("linear regression_v1")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.linear_regression()

        # linear regression advanced
        # print("advanced linear regression v1")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.linear_regression_advanced()


        # random forest regressor
        # print("random forest regressor 5")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.random_forest_regressor()



        # extra random forest regressor
        # print("extra random forest regressor")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.extra_tree_regressor()



        # gradient boosting
        # print("gradient boosting")
        # x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.gradient_boosting_regressor()

        # error
        # total_error = self.total_MSE(x_train_predicts, self.master_lat, y_train_predicts, self.master_long)

        # final learner
        # best random forest regressor
        print("best random forest regressor")
        x_train_predicts, y_train_predicts, x_test_predicts, y_test_predicts = self.best_random_forest_regressor()

        # submission
        new_file = open("./data/submission_best_random_forest.txt", "w")

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

    def gradient_boosting_regressor(self):

        learner = GradientBoostingRegressor()
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        learner.fit(self.master_features, self.master_long)

        long_train_preds = learner.predict(self.master_features)
        long_test_preds = learner.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def extra_tree_regressor(self):

        learner = ExtraTreesRegressor(max_depth=10)
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        learner.fit(self.master_features, self.master_long)

        long_train_preds = learner.predict(self.master_features)
        long_test_preds = learner.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def best_random_forest_regressor(self):

        learner = RandomForestRegressor(n_estimators=90, max_depth=9) # best parameters for lat
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        # cross validation score
        score_lat = cross_val_score(learner, self.master_features, self.master_lat, cv=5)
        print(f"score lat is {score_lat.mean()}")

        learner = RandomForestRegressor(n_estimators=100, max_depth=10) # best parameters for long
        learner.fit(self.master_features, self.master_long)

        long_train_preds = learner.predict(self.master_features)
        long_test_preds = learner.predict(self.master_test_features)

        # cross validation score
        score_long = cross_val_score(learner, self.master_features, self.master_long, cv=5)
        print(f"score long is {score_long.mean()}")

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def random_forest_regressor(self):

        learner = RandomForestRegressor(max_depth=5)
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        learner.fit(self.master_features, self.master_long)

        long_train_preds = learner.predict(self.master_features)
        long_test_preds = learner.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def linear_regression(self):

        learner = LinearRegression(n_jobs=17)
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        learner.fit(self.master_features, self.master_long)

        long_train_preds = learner.predict(self.master_features)
        long_test_preds = learner.predict(self.master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def linear_regression_advanced(self):

        learner = LinearRegression(n_jobs=17)
        learner.fit(self.master_features, self.master_lat)

        lat_train_preds = learner.predict(self.master_features)
        lat_test_preds = learner.predict(self.master_test_features)

        # find shape
        n, m = self.raw_master_features.shape

        # predict train longitude by using new master features
        new_master_train_features = np.hstack((self.master_features, lat_train_preds.reshape((n, 1))))
        new_train_scaler = MinMaxScaler()
        new_train_scaler.fit(new_master_train_features)
        new_train_scaler.transform(new_master_train_features)

        learner.fit(new_master_train_features, self.master_long)

        long_train_preds = learner.predict(new_master_train_features)

        # new shape
        n, m = self.master_test_features.shape

        # predict train longitude by using new master features
        new_master_test_features = np.hstack((self.master_test_features, lat_test_preds.reshape((n, 1))))
        new_test_scaler = MinMaxScaler()
        new_test_scaler.fit(new_master_test_features)
        new_test_scaler.transform(new_master_test_features)

        long_test_preds = learner.predict(new_master_test_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def kNN_norm(self, test):

        # kNN
        kNN = KNeighborsRegressor(n_neighbors=9, n_jobs= 2)

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

        SVM = SVR(kernel = "linear", max_iter = 1000)

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



    def runGridSearch_random_forest(self):

        params = [{'n_estimators': [10, 30, 50, 70, 90, 100], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None], 'n_jobs': [17]}]

        learner = RandomForestRegressor(n_jobs=17)

        gs = GridSearchCV(learner, params, 'neg_mean_squared_error', cv=5, n_jobs = 15)
        gs.fit(self.master_features, self.master_lat)


        print("The best lat parameters found were: ")
        print(gs.best_params_)
        print(gs.get_params())

        lat_test_preds = gs.predict(self.master_test_features)
        lat_train_preds = gs.predict(self.master_features)

        gs.fit(self.master_features, self.master_long)
        print("the best long params for longitude were: ")
        print(gs.best_params_)
        print(gs.get_params())

        long_test_preds = gs.predict(self.master_test_features)
        long_train_preds = gs.predict(self.master_features)

        return lat_train_preds, long_train_preds, lat_test_preds, long_test_preds

    def bagging(self):
        learner = DecisionTreeRegressor(max_depth = 5)
        bag = BaggingRegressor(learner, n_jobs = 2)
        bag.fit(self.master_features, self.master_lat)


        lat_test_predicts = bag.predict(self.master_test_features)

        lat_train_predicts = bag.predict(self.master_features)

        bag.fit(self.master_features, self.master_long)

        long_test_predicts = bag.predict(self.master_test_features)

        long_train_predicts = bag.predict(self.master_features)

        return lat_train_predicts, long_train_predicts, lat_test_predicts, long_test_predicts



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

        # v2 features
        pickle_in_5 = open("./data/master_features_v2.pkl", "rb")
        self.master_features_v2, desc5 = pickle.load(pickle_in_5)
        print(desc)

        pickle_in6 = open("./data/master_test_features_v2.pkl", "rb")
        self.master_test_features_v2, desc6 = pickle.load(pickle_in6)
        print(desc6)

        self.master_features_v2 = np.array(self.master_features_v2)
        self.master_test_features_v2 = np.array(self.master_test_features_v2)
        self.raw_master_features_v2 = self.master_features_v2

        self.scaler_v2 = MinMaxScaler()
        self.scaler_v2.fit(self.master_features_v2)
        self.master_features_v2 = self.scaler_v2.transform(self.master_features_v2)
        self.master_test_features_v2 = self.scaler_v2.transform(self.master_features_v2)

    def get_target_array(self, target_dict):

        target_list = list(target_dict.values())
        return np.array(target_list)


if __name__ == "__main__":

    myModelSelection = modelSelection()
    myModelSelection.start()