import pickle
from multiprocessing import Pool
from random import randint
def process(train_set, graph):
    master_list = []

    for key,value in train_set:
        temp_list = []
        #Hour1, Hour2, Hour3, Lat, Lon, Posts
        #not including latitude and longitude
        indeces = [0,1,2,5]
        for num in indeces:
            temp_list.append(value[num])

        #number of friends
        num_friends = len(graph[key])
        temp_list.append(num_friends)

        #most common latitude among friends - not working yet
        friends_lat = friends_top_lat(key, train_set, graph)
        temp_list.append(friends_lat)

        #most common latitude among friends based on similarity from their top hours
        lat_from_hour = top_lat_hour(key, train_set, graph)
        temp_list.append(lat_from_hour)



def friends_top_lat(id, train_set, graph):
    friends = graph[id]
    #no friends return -1
    if len(friends) == 0:
        return -1
    your_data = train_set[id]
    location_dict= dict()

    for f in friends:
        their_data = train_set[f]
        lat = their_data[3]
        if lat in location_dict:
            location_dict[lat] = location_dict[lat] + 1
        else:
            location_dict[lat] = 1

    #in the event of a tie it picks the first max encountered MUST FIX
    #max_loc = max(location_dict.keys(), key=(lambda k: location_dict[k]))

    max_value = max(location_dict.values())
    print("max based on top location (value and then location")
    print(max_value)

    max_keys = [k for k,v in location_dict.items() if v == max_value]
    print(max_keys)
    if len(max_keys) == 1:
        return max_keys[0]
    else:
        return max_keys

def top_lat_hour(id, train_set, graph):
    #return location with closest hour1, then hour2, then hour3
    friends = graph[id]
    # no friends return -1
    if len(friends) == 0:
        return -1
    your_data = train_set[id]
    key_hour = []

    for i in range(3):
        print(f"AT HOUR {i}")
        # dictionary where id is key and the values are the differences between the points' hours as a list
        hour_dict = dict()
        for f in friends:
            #checking if in training set
            if f not in train_set:
                continue
            their_data = train_set[f]
            if their_data[i] != 25 and your_data[i] != 25:
                print(your_data[i])
                print(their_data[i])
                hour_dict[f] = abs(your_data[i] - their_data[i])
            else:
                #25 means they cannot be compared
                hour_dict[f] = 25

        print(hour_dict)
        min_hour1_diff = min(hour_dict.values())
        print(min_hour1_diff)
        key_hour = [k for k,v in hour_dict.items() if v == min_hour1_diff]
        print(key_hour)

        #return location whose hour1 is closest to the point's hour1 if there is only 1 min value
        if len(key_hour) == 1 and key_hour != 25:
            print("hi")
            return train_set[key_hour[0]][3]

    #if there is still a tie pick randomly

    index = randint(len(key_hour))
    key = key_hour[index]
    return train_set[key][3]






if __name__ == "__main__":
    graph_in = open("./data/master_graph_dict.pkl", "rb")
    graph, desc = pickle.load(graph_in)

    train_in = open("./data/posts_train_dict.pkl", "rb")
    train, desc2 = pickle.load(train_in)

    test_in = open("./data/posts_test_dict.pkl", "rb")
    test, desc3 = pickle.load(test_in)

    print(len(graph))
    print(len(train))

    print(top_lat_hour(7829, train, graph))



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
