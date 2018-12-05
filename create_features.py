import pickle
from multiprocessing import Pool
def process(id, train_info, friends):
    temp_list = []

    for num in train_info:
        temp_list.append(num)



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
    l = []
    for key,value in train:
        l.append((key, value, graph[key]))

    p = Pool(processes=4)
    processed = p.starmap(process, l)
    pickle_desc = "array of extrated features for every point in the training set"
    pickle_out = open('./data/processed_train.pkl', 'wb')
    pickle.dump((processed, pickle_desc), pickle_out)
    pickle_out.close()
    p.close()
