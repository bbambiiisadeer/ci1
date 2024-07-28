from random import randrange
def data_split (data, folds):
    splitted_data = list()
    ori_data = list(data)
    fold_size = int(len(data) / folds)
    for i in range (folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(ori_data))
            fold.append(ori_data.pop(index))
        splitted_data.append(fold)
    return splitted_data
