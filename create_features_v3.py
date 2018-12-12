import pickle
from multiprocessing import Pool
from random import randint
import numpy as np
import statistics
from math import sqrt

'''
Creates master features for test set.
'''

class createFeatures(object):

    def __init__(self):

        self.train_set = None
        self.test_set = None
        self.graph = None
        self.rand_ids = None

        self.master_features = None

    def start(self):

        self.initialize_values()

        self.master_features = [None]*len(self.test_set)

        counter = 0
        for key, value in self.test_set.items():

            temp_list = [None]*26 #initialize

            temp_list[0] = value[0] #hour1 and importance
            if value[0] == 25:
                temp_list[1] = 0
            else:
                temp_list[1] = 1

            temp_list[2] = value[1] #hour2 and importance
            if value[1] == 25:
                temp_list[3] = 0
            else:
                temp_list[3] = 1

            temp_list[4] = value[2] #hour3 and importance
            if value[2] == 25:
                temp_list[5] = 0
            else:
                temp_list[5] = 1

            temp_list[6] = value[3] #number of posts

            if key not in self.graph: #number of friends
                temp_list[7] = 0
            else:
                num_friends = len(self.graph[key])
                temp_list[7] = num_friends

            friends_lat, friends_long, importancefc = self.friends_cluster_location(key)
            temp_list[8] = friends_lat #location where friends cluter
            temp_list[9] = friends_long
            temp_list[10] = importancefc

            #most common location among friends based on similarity from their top hours (can return -1) starting with hour1
            lat_from_hour1, long_from_hour1, importance_hour1 = self.loc_from_hours(key, 0)
            temp_list[11] = lat_from_hour1
            temp_list[12] = long_from_hour1
            temp_list[13] = importance_hour1

            #starting with hour2
            lat_from_hour2, long_from_hour2, importance_hour2 = self.loc_from_hours(key, 1)
            temp_list[14] = lat_from_hour2
            temp_list[15] = long_from_hour2
            temp_list[16] = importance_hour2

            #starting with hour3
            lat_from_hour3, long_from_hour3, importance_hour3 = self.loc_from_hours(key, 2)
            temp_list[17] = lat_from_hour3
            temp_list[18] = long_from_hour3
            temp_list[19] = importance_hour3

            # location from friends whose number of posts is most similar to their number of posts
            lat_from_posts, long_from_posts, importance_posts = self.loc_from_posts(key)
            temp_list[20] = lat_from_posts
            temp_list[21] = long_from_posts
            temp_list[22] = importance_posts

            # location from friends whose hours is most similar to their hours
            lat_from_all_hours, long_from_all_hours, importance_all_hours = self.loc_from_all_hours(key)
            temp_list[23] = lat_from_all_hours
            temp_list[24] = long_from_all_hours
            temp_list[25] = importance_all_hours

            self.master_features[counter] = temp_list
            counter += 1

            if counter%1000 == 0:
                print(f"\ncounter at {counter}")
                print(key)
                print(temp_list)

        pickle_out = open("./data/master_test_features.pkl", "wb")
        desc = "array of all test users and their features; see doc for details"
        pickle.dump((self.master_features, desc), pickle_out)
        pickle_out.close()

    def initialize_values(self):

        graph_in = open("./data/master_graph_dict.pkl", "rb")
        graph, desc = pickle.load(graph_in)
        self.graph = graph

        train_in = open("./data/posts_train_dict.pkl", "rb")
        train, desc2 = pickle.load(train_in)
        self.train_set = train

        test_in = open("./data/posts_test_dict.pkl", "rb")
        test, desc3 = pickle.load(test_in)
        self.test_set = test

        group_in = open("./data/id_by_group.pkl", "rb")
        (rand_ids, null_islanders, hour1, hour2, hour3), desc = pickle.load(group_in)
        self.rand_ids = rand_ids

    def friends_cluster_location(self, id):

        # check if user exists in graph and if he/she has friends
        if id not in self.graph:
            return -1, -1, 0
        friends = self.graph[id]
        if len(friends) == 0:
            return -1, -1, 0

        # note: std of location is 70
        # print(f"proceeding with {len(friends)} friends")

        r = 40
        friends_pop = dict()
        max_popularity = 0
        for f in friends:

            if f not in self.train_set:
                continue

            f_x = self.train_set[f][3]
            f_y = self.train_set[f][4]

            # print("start with friend")
            # print(f_x)
            # print(f_y)
            # print()

            temp_count = 0
            for fr in friends:
                if fr not in self.train_set:
                    continue
                fr_x = self.train_set[fr][3]
                fr_y = self.train_set[fr][4]

                # print((((f_x-fr_x)**2) + ((f_y-fr_y)**2))**(1/2))

                if (((f_x - fr_x) ** 2) + ((f_y - fr_y) ** 2)) ** (1 / 2) <= r:
                    temp_count += 1

            friends_pop[f] = temp_count
            if temp_count > max_popularity:
                max_popularity = temp_count

        # print(friends_pop)
        # print(max_popularity)
        most_popular_friends = [friend for friend, popularity in friends_pop.items() if popularity == max_popularity]
        # print(most_popular_friends)
        if len(most_popular_friends) == 1:
            index = most_popular_friends[0]
            friend_x, friend_y = self.train_set[index][3], self.train_set[index][4]
            return friend_x, friend_y, 1
        elif len(most_popular_friends) > 1:
            index = np.random.choice(most_popular_friends, 1)[0]
            friend_x, friend_y = self.train_set[index][3], self.train_set[index][4]
            return friend_x, friend_y, 1
        else:
            # print(f"User {id} has no friend clusters")
            return -1, -1, 0

    def loc_from_hours(self, id, start):

        # return location with closest hour1, then hour2, then hour3
        # no friends return -1,-1 for loc and 0 for importance
        if id not in self.graph:
            return -1, -1, 0
        friends = self.graph[id]
        # no friends return -1,-1 for loc and 0 for importance
        if len(friends) == 0:
            return -1, -1, 0
        your_data = self.test_set[id]
        key_hour = []

        for i in range(start, 3):
            # print(f"AT HOUR {i}")
            # dictionary where id is key and the values are the differences between the points' hours as a list
            hour_dict = dict()
            for f in friends:
                # checking if in training set
                if f not in self.train_set:
                    continue
                their_data = self.train_set[f]
                # do not put ids with value 25 into the dictionary
                if their_data[i] != 25 and your_data[i] != 25:
                    hour_dict[f] = abs(your_data[i] - their_data[i])
            if len(hour_dict) == 0:
                continue
            min_hour1_diff = min(hour_dict.values())
            key_hour = [k for k, v in hour_dict.items() if v == min_hour1_diff]
            # print(key_hour)
            # return location whose hour1 is closest to the point's hour1 if there is only 1 min value
            if len(key_hour) == 1 and key_hour != 25:
                return self.train_set[key_hour[0]][3], self.train_set[key_hour[0]][4], 1
            else:
                # now only checking the ids who have the minimum hour1
                # either you or all your friends have no valid hours
                if len(key_hour) != 0:
                    friends = key_hour

        # if there are no valid hours
        if len(key_hour) == 0:
            return -1, -1, 0
        # if there is still a tie pick randomly
        key = np.random.choice(np.array(key_hour), 1)[0]
        return self.train_set[key][3], self.train_set[key][4], 1

    # return location of friend most similar to user based on number of posts
    def loc_from_posts(self, id):
        if id not in self.graph:
            return -1, -1, 0

        friends = self.graph[id]
        if len(friends) == 0:
            return -1, -1, 0
        your_data = self.test_set[id]
        posts_dict = dict()

        for f in friends:
            # don't consider points without locations
            if f not in self.train_set:
                continue
            their_data = self.train_set[f]
            # 5 is index of number of posts
            diff = abs(your_data[3] - their_data[5])
            posts_dict[f] = diff

        if len(posts_dict) != 0:
            min_diff = min(posts_dict.values())
            min_keys = [k for k, v in posts_dict.items() if v == min_diff]

            # return location whose num of posts is closest to the point's num of posts if there is only 1 min value
            if len(min_keys) == 1:
                key = min_keys[0]
                return self.train_set[key][3], self.train_set[key][4], 1
            else:
                # random
                key = np.random.choice(np.array(min_keys), 1)[0]
                return self.train_set[key][3], self.train_set[key][4], 1
        else:
            return -1,-1,0

    def loc_from_all_hours(self, id):
        if id not in self.graph:
            return -1, -1, 0

        friends = self.graph[id]
        if len(friends) == 0:
            return -1, -1, 0

        your_data = self.test_set[id]
        hours_dict = dict()

        for f in friends:
            # don't consider points without locations
            if f not in self.train_set:
                continue
            their_data = self.train_set[f]
            sum = 0
            # boolean to make sure the sum is not 0 because all the values were 25
            real_sum = False
            for i in range(3):
                if your_data[i] == 25 or their_data[i] == 25:
                    continue
                diff_sqr = (your_data[i] - their_data[i]) ** 2
                sum += diff_sqr
                real_sum = True
            # if we should consider the sum, then add the difference to the dictionary
            if real_sum:
                total_diff_sqrt = sqrt(sum)
                hours_dict[f] = total_diff_sqrt
        # no friends with valid hours or you don't have valid hours
        if len(hours_dict) == 0:
            return -1, -1, 0
        # print(hours_dict)
        min_diff = min(hours_dict.values())
        min_keys = [k for k, v in hours_dict.items() if v == min_diff]

        # return location whose hours are most similar to theirs
        if len(min_keys) == 1:
            key = min_keys[0]
            return self.train_set[key][3], self.train_set[key][4], 1
        else:
            # random
            key = np.random.choice(np.array(min_keys), 1)[0]
            return self.train_set[key][3], self.train_set[key][4], 1


if __name__ == "__main__":

    ourFeatures = createFeatures()
    ourFeatures.start()