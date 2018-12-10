import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error

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

        print("norm")
        x_predicts, y_predicts = self.kNN_norm(False)
        self.total_MSE(x_predicts, self.master_lat, y_predicts, self.master_long)
        print("advanced")
        x_predicts, y_predicts = self.kNN_advanced()
        self.total_MSE(x_predicts, self.master_lat, y_predicts, self.master_long)

        new_file = open("./data/submission_knn3_normal.txt", "w")

        counter = 0
        new_file.write("Id,Lat,Lon")
        for key in self.test_dict.keys():
            string = "\n" + str(key) + "," + str(x_predicts[counter]) + "," + str(y_predicts[counter])
            new_file.write(string)
            counter += 1
        new_file.close()

    def total_MSE(self, pred_x, targ_x, pred_y, targ_y):

        location = pred_x + pred_y
        pred_location = targ_x + targ_y

        loss = (mean_squared_error(location, pred_location)) ** (1 / 2)

        print("Total MSE " + str(loss))

    def kNN_norm(self, test):

        # kNN
        kNN = KNeighborsRegressor(n_neighbors=5, n_jobs= 7)

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
        kNN = KNeighborsRegressor(n_neighbors=5, n_jobs= 7)

        kNN.fit(self.master_features, self.master_lat)
        lat_predicts = kNN.predict(self.master_features)

        n, m = self.raw_master_features.shape
        new_master_features = np.hstack((self.master_features, lat_predicts.reshape((n, 1))))

        new_scaler = MinMaxScaler()
        new_scaler.fit(new_master_features)
        new_master_features = new_scaler.transform(new_master_features)

        kNN.fit(new_master_features, self.master_long)
        long_predicts = kNN.predict(new_master_features)

        return lat_predicts, long_predicts

        # kNN_parameters = [{'n_neighbors': [1, 2, 3, 4, 5],
        #                    'weights': ['uniform', 'distance'],
        #                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        #                    'p': [1, 2],
        #                    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'manhalanobis']
        #                    }]
        #
        # kNN_gs = GridSearchCV(estimator=kNN, param_grid=kNN_parameters, scoring='neg_mean_squared_error', n_jobs=2, cv=5)
        # kNN_gs.fit(X=self.master_features, y=self.master_lat)
        # print(kNN_gs.cv_results_)
        # knn_results = pd.DataFrame(data=kNN_gs.cv_results_)
        # print(knn_results)

        #SVC

        #Decision Tree

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