import numpy as np

f = open("./data/test_file.txt", "w")

my_list = np.arange(0, 12)

for number in my_list:
    f.write(str(number))
    f.write("\n")

f.close()