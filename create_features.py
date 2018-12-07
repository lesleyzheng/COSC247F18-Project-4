import pickle
from multiprocessing import Pool
from random import randint
import numpy as np
import statistics
from math import sqrt

def process(train_set, graph):
    master_list = []

    for key,value in train_set.items():
        temp_list = []

        # Adding hour1, 2, 3, and num of posts
        indeces = [0,1,2,5]
        for num in indeces:
            temp_list.append(value[num])
            #importance (num of posts does not have an importance)
            if num != 5:
                if num == 25:
                    temp_list.append(0)
                else:
                    temp_list.append(1)

        # Add number of friends
        if key not in graph:
            temp_list.append(0)
        else:
            num_friends = len(graph[key])
            temp_list.append(num_friends)

        # TB fixed: finding position with most friends around it
        friends_lat, friends_long, importancefc = friends_median_lat(key, train_set, graph)
        temp_list.append(friends_lat)
        temp_list.append(friends_long)
        temp_list.append(importancefc)

        #most common latitude among friends based on similarity from their top hours (can return -1) starting with hour1
        lat_from_hour1, long_from_hour1, importance_hour1 = loc_from_hours(key, train_set, graph, 0)
        temp_list.append(lat_from_hour1)
        temp_list.append(long_from_hour1)
        temp_list.append(importance_hour1)

        # most common latitude among friends based on similarity from their top hours (can return -1) starting with hour2
        lat_from_hour2, long_from_hour2, importance_hour2 = loc_from_hours(key, train_set, graph, 1)
        temp_list.append(lat_from_hour2)
        temp_list.append(long_from_hour2)
        temp_list.append(importance_hour2)

        # most common latitude among friends based on similarity from their top hours (can return -1) starting with hour3
        lat_from_hour3, long_from_hour3, importance_hour3 = loc_from_hours(key, train_set, graph, 2)
        temp_list.append(lat_from_hour3)
        temp_list.append(long_from_hour3)
        temp_list.append(importance_hour3)

        #location from friends whose number of posts is most similar to their number of posts
        lat_from_posts, long_from_posts, importance_posts = loc_from_posts(key, train_set, graph)
        temp_list.append(lat_from_posts)
        temp_list.append(long_from_posts)
        temp_list.append(importance_posts)


        #location from friends whose hours is most similar to their hours
        lat_from_all_hours, long_from_all_hours, importance_all_hours = loc_from_all_hours(key, train_set, graph)
        temp_list.append(lat_from_all_hours)
        temp_list.append(long_from_all_hours)
        temp_list.append(importance_all_hours)

        master_list.append(temp_list)
    return master_list

def friends_cluster_location(id, train_set, graph):

    # check if user exists in graph and if he/she has friends
    if id not in graph:
        return -1, -1, 0
    friends = graph[id]
    if len(friends) == 0:
        return -1, -1, 0

    #note: std of location is 70
    print(f"proceeding with {len(friends)} friends")

    r = 40
    friends_pop = dict()
    max_popularity = 0
    for f in friends:

        if f not in train_set:
            continue

        f_x = train_set[f][3]
        f_y = train_set[f][4]

        # print("start with friend")
        # print(f_x)
        # print(f_y)
        # print()

        temp_count = 0
        for fr in friends:
            if fr not in train_set:
                continue
            fr_x = train_set[fr][3]
            fr_y = train_set[fr][4]

            # print((((f_x-fr_x)**2) + ((f_y-fr_y)**2))**(1/2))

            if (((f_x-fr_x)**2) + ((f_y-fr_y)**2))**(1/2) <= r:
                temp_count += 1

        friends_pop[f] = temp_count
        if temp_count > max_popularity:
            max_popularity = temp_count

    print(friends_pop)
    print(max_popularity)
    most_popular_friends = [friend for friend, popularity in friends_pop.items() if popularity == max_popularity]
    print(most_popular_friends)
    if len(most_popular_friends) == 1:
        index = most_popular_friends[0]
        friend_x, friend_y = train_set[index][3], train_set[index][4]
        return friend_x, friend_y, 1
    elif len(most_popular_friends) > 1:
        index = np.random.choice(most_popular_friends, 1)[0]
        friend_x, friend_y = train_set[index][3], train_set[index][4]
        return friend_x, friend_y, 1
    else:
        print(f"User {id} has no friend clusters")
        return -1, -1, 0

def loc_from_hours(id, train_set, graph, start):

    #return location with closest hour1, then hour2, then hour3
    # no friends return -1,-1 for loc and 0 for importance
    if id not in graph:
        return -1, -1, 0
    friends = graph[id]
    # no friends return -1,-1 for loc and 0 for importance
    if len(friends) == 0:
        return -1, -1, 0
    your_data = train_set[id]
    key_hour = []

    for i in range(start, 3):
        print(f"AT HOUR {i}")
        # dictionary where id is key and the values are the differences between the points' hours as a list
        hour_dict = dict()
        for f in friends:
            #checking if in training set
            if f not in train_set:
                continue
            their_data = train_set[f]
            #do not put ids with value 25 into the dictionary
            if their_data[i] != 25 and your_data[i] != 25:
                hour_dict[f] = abs(your_data[i] - their_data[i])
        min_hour1_diff = min(hour_dict.values())
        key_hour = [k for k, v in hour_dict.items() if v == min_hour1_diff]
        print(key_hour)
        #return location whose hour1 is closest to the point's hour1 if there is only 1 min value
        if len(key_hour) == 1 and key_hour != 25:
            return train_set[key_hour[0]][3], train_set[key_hour[0]][4], 1
        else:
            #now only checking the ids who have the minimum hour1
            # either you or all your friends have no valid hours
            if len(key_hour) != 0:
                friends = key_hour

    #if there are no valid hours
    if len(key_hour) == 0:
        return -1,-1,0
    #if there is still a tie pick randomly
    index = randint(len(key_hour))
    key = key_hour[index]
    return train_set[key][3], train_set[key][4], 1

#return location of friend most similar to user based on number of posts
def loc_from_posts(id, train_set, graph):
    if id not in graph:
        return -1,-1, 0

    friends = graph[id]
    if len(friends) == 0:
        return -1, -1, 0
    your_data = train_set[id]
    posts_dict = dict()

    for f in friends:
        #don't consider points without locations
        if f not in train_set:
            continue
        their_data = train_set[f]
        #5 is index of number of posts
        diff = abs(your_data[5] - their_data[5])
        posts_dict[f] = diff

    min_diff = min(posts_dict.values())
    min_keys = [k for k,v in posts_dict.items() if v == min_diff]

    # return location whose num of posts is closest to the point's num of posts if there is only 1 min value
    if len(min_keys) == 1:
        key = min_keys[0]
        return train_set[key][3], train_set[key][4], 1
    else:
        #random
        index = randint(len(min_keys))
        key = min_keys[index]
        return train_set[key][3], train_set[key][4], 1

def loc_from_all_hours(id, train_set, graph):
    if id not in graph:
        return -1, -1, 0

    friends = graph[id]
    if len(friends) == 0:
        return -1, -1, 0

    your_data = train_set[id]
    hours_dict = dict()

    for f in friends:
        # don't consider points without locations
        if f not in train_set:
            continue
        their_data = train_set[f]
        sum = 0
        #boolean to make sure the sum is not 0 because all the values were 25
        real_sum = False
        for i in range(3):
            if your_data[i] == 25 or their_data[i] == 25:
                continue
            diff_sqr = (your_data[i] - their_data[i]) ** 2
            sum += diff_sqr
            real_sum = True
        #if we should consider the sum, then add the difference to the dictionary
        if real_sum:
            total_diff_sqrt = sqrt(sum)
            hours_dict[f] = total_diff_sqrt
    #no friends with valid hours or you don't have valid hours
    if len(hours_dict) == 0:
        return -1,-1, 0
    print(hours_dict)
    min_diff = min(hours_dict.values())
    min_keys = [k for k, v in hours_dict.items() if v == min_diff]

    # return location whose hours are most similar to theirs
    if len(min_keys) == 1:
        key = min_keys[0]
        return train_set[key][3], train_set[key][4], 1
    else:
        # random
        index = randint(len(min_keys))
        key = min_keys[index]
        return train_set[key][3], train_set[key][4], 1



def no_friends(graph, train_set):
    count = 0
    for k, v in train_set.items():
        if k not in graph:
            print(k)
            count += 1
    print(count)

def max_friends(graph):
    max = 0
    for k,v in graph.items():
        if len(v) > max:
            max = len(v)
    print(max)



if __name__ == "__main__":
    graph_in = open("./data/master_graph_dict.pkl", "rb")
    graph, desc = pickle.load(graph_in)

    train_in = open("./data/posts_train_dict.pkl", "rb")
    train, desc2 = pickle.load(train_in)

    test_in = open("./data/posts_test_dict.pkl", "rb")
    test, desc3 = pickle.load(test_in)

    print(len(graph))
    print(len(train))


    print(loc_from_all_hours(2, train, graph))
    print(train[8100])

    # print(friends_median_hour1(3, train, graph))

    group_in = open("./data/id_by_group.pkl", "rb")
    (rand_ids, null_islanders, hour1, hour2, hour3), desc = pickle.load(group_in)

    for ID in rand_ids:
        print(f"\nID {ID}")
        print(friends_cluster_location(ID, train, graph))




    # THIS IS NOT WORKING BECAUSE YOU NEED WHOLE GRAPH AND TRAINING
    # l = []
    # for key,value in train:
    #     l.append((key, value, graph))
    #
    # p = Pool(processes=4)
    # processed = p.starmap(process, l)
    # pickle_desc = "array of extrated features for every point in the training set"
    # pickle_out = open('./data/processed_train.pkl', 'wb')
    # pickle.dump((processed, pickle_desc), pickle_out)
    # pickle_out.close()
    # p.close()
