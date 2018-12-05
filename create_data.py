import pickle
def create_graph():
    #graph where key is id and value is a list of their friends ids
    master_dict = dict()
    graph = open("./data/graph.txt", "r")
    lines = graph.readlines()
    for line in lines:
        nums = line.split()
        key = int(nums[0])
        value = int(nums[1])
        if key in master_dict:
            friends = master_dict[key]
            friends.append(value)
            master_dict[key] = friends

        else:
            master_dict[key] = [value]


    pickle_out = open("./data/master_graph_dict.pkl", 'wb')
    desc = "a dictionary of the social graph where the keys are ids and the values are a list of their friends ids"
    pickle.dump((master_dict, desc), pickle_out)
    pickle_out.close()




if __name__ == '__main__':
    create_graph()