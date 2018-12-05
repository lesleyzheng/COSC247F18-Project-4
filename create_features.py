import pickle
from multiprocessing import Pool
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

        #most common latitude among friends
        friends_lat = friends_top_lat(key, train_set, graph)
        temp_list.append(friends_lat)



def friends_top_lat(id, train_set, graph):
    friends = graph[id]

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

    max_keys = [k for k,v in location_dict.items() if v == max_value]
    if len(max_keys) == 1:
        return max_keys[0]

    #but else do what??

    return max_loc






if __name__ == "__main__":
    graph_in = open("./data/master_graph_dict.pkl", "rb")
    graph, desc = pickle.load(graph_in)

    train_in = open("./data/posts_train_dict.pkl", "rb")
    train, desc2 = pickle.load(train_in)

    test_in = open("./data/posts_test_dict.pkl", "rb")
    test, desc3 = pickle.load(test_in)

    print(len(graph))
    print(len(train))

    #number of friends, mod of location, mod of hour, "most similar location based on hour",

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
