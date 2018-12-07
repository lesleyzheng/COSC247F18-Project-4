'''

6.	Data Cleaning
a.	Exclude Null Islands
i.	First see % of total data set  reflect % of test set
ii.	There could be a certain set of characteristics for null islanders
b.	Check if all users exist in dataset – check if user id follows a counter - NO
c.	Check each user has all valid features
i.	What to do with null island?
ii.	What to do with 25?
1.	How should we include it?
a.	Additional features indicating whether original features have valid inputs
d.	Outlier detection
i.	Pca/clustering/general – centrally xxx
1.	Give me distance from each point; plot and get rid of obvious outliers
2.	Could just plot
e.	What to do with users with no connections
i.	Could give some indication

1.	First predict latitude, then predict longitude, and vice versa
2.	Predict latitude, then include latitude as a feature, then predict longitude
3.	Reverse that


'''

import numpy as np
import pickle


class cleaning(object):

    def __init__(self):

        self.users = []
        self.unique_ids = set()
        self.num_users = 0
        self.null_islanders = set()
        self.miss_features = set()

        #hours stuff
        self.hour1 = set()
        self.hour2 = set()
        self.hour3 = set()

    def start(self):

        f = open("./data/posts_train.txt", "r")

        first_line = True
        for line in f:

            if first_line:
                first_line = False
            else:
                user_info = line.split(",")
                self.unique_ids.add(user_info[0])
                self.all_features(user_info)
                self.null_islander(user_info)
                self.invalid_hours(user_info)
                self.users.append(user_info)

        print(f"There are {len(self.unique_ids)} unique IDs.")
        print(f"There are {len(self.users)} number of users.")
        print(f"There are {len(self.null_islanders)} number of null islands.")
        print(f"There are {len(self.hour1)} number of invalid hour1s, {len(self.hour2)} number of invalid hour2s "
              f"and {len(self.hour3)} number of invalid hour3s.")
        print(f"There are {len(self.miss_features)} people who miss features.")

        print(f"Intersections")
        print(len(self.null_islanders.intersection(self.hour1)))
        print(len(self.hour1.intersection(self.hour2)))
        print(len(self.hour2.intersection(self.hour3)))
        print(len(self.hour1.intersection(self.hour3)))
        print(len(self.hour1.intersection(self.hour2).intersection(self.hour3)))
        print(len(self.hour1.intersection(self.hour2).intersection(self.hour3).intersection(self.null_islanders)))

        # self.null_islanders = set()
        # self.miss_features = set()
        #
        # # hours stuff
        # self.hour1 = set()
        # self.hour2 = set()
        # self.hour3 = set()

        pickle_out = open("./data/id_by_group.pkl", 'wb')
        desc = "list of arrays: null_islanders, hour1, hour2, hour3"
        pickle.dump(((list(self.null_islanders), list(self.hour1), list(self.hour2), list(self.hour3)), desc), pickle_out)
        pickle_out.close()

    def all_features(self, user_array):

        if len(user_array) != 7:
            print(f"Missing features for {user_array[0]}")
            self.miss_features.add(user_array[0])

    def null_islander(self, user_array):

        # Id,Hour1,Hour2,Hour3,Lat,Lon,Posts
        if float(user_array[4]) == 0.0 and float(user_array[5]) == 0.0:
            self.null_islanders.add(user_array[0])

    def invalid_hours(self, user_array):

        if int(user_array[1]) == 25:
            self.hour1.add(user_array[0])
        if int(user_array[2]) == 25:
            self.hour2.add(user_array[0])
        if int(user_array[3]) == 25:
            self.hour3.add(user_array[0])

        for i in range(1,4):
            # print(int(user_array[i]))
            # print(np.arange(1, 26))
            # print(int(user_array[i]) not in np.arange(1, 26))

            if int(user_array[i]) not in np.arange(0, 24) and int(user_array[i]) != 25:
                print(f"{user_array[0]} has an hour out of range")
                print(int(user_array[i]))


if __name__ == "__main__":

    myClean = cleaning()
    myClean.start()
