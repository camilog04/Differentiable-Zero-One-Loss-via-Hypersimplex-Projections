import random

with open("HIGGS", "r") as fin, \
     open("higgs.train", "w") as ftrain, \
     open("higgs.valid", "w") as fvalid:

    for line in fin:
        if random.random() < 0.8:
            ftrain.write(line)
        else:
            fvalid.write(line)