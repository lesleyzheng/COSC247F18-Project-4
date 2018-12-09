import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

class modelSelection(object):

    def __init__(self):

        self.master_lat = None
        self.master_long = None
        self.master_loc = None

        self.master_features = None
        self.master_test_features = None


    def start(self):

        self.initializeValues()

        self.gridSearch()

    def gridSearch(self):

        #kNN 1-5
        kNN = KNeighborsRegressor()

        kNN_parameters = [{'n_neighbors': [1, 2, 3, 4, 5],
                           'weights': ['uniform', 'distance'],
                           'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                           'p': [1, 2],
                           'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'manhalanobis']
                           }]

        kNN_gs = GridSearchCV(estimator=kNN, param_grid=kNN_parameters, scoring='neg_mean_squared_error', n_jobs=2, cv=5)
        kNN_gs.fit(X=self.master_features, y=self.master_lat)
        print(kNN_gs.cv_results_)
        knn_results = pd.DataFrame(data=kNN_gs.cv_results_)
        print(knn_results)

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

        # convert to np array
        self.master_features = np.array(self.master_features)
        self.master_test_features = np.array(self.master_test_features)
        self.master_lat = self.get_target_array(self.master_lat)
        self.master_long = self.get_target_array(self.master_long)
        self.master_loc = self.get_target_array(self.master_loc)

        # scaleValues
        scaler = MinMaxScaler()
        scaler.fit(self.master_features)
        self.master_features = scaler.transform(self.master_features)
        self.master_test_features = scaler.transform(self.master_test_features)

    def get_target_array(self, target_dict):

        target_list = list(target_dict.values())
        return np.array(target_list)


if __name__ == "__main__":

    myModelSelection = modelSelection()
    myModelSelection.start()