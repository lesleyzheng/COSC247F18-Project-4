import pickle

def process():
    length = len(train)
    train_set = [[None]] * length



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
