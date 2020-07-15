
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

NUM_LINES = 891
FARE = 0
FAMILY = 1
AGE = 2
GENDER = 3

def split_comma(line):
    ptr = 0
    buffer = []
    in_quot = 0
    ret = []
    
    while(ptr < len(line)):
        if(line[ptr] == ',' and in_quot % 2 == 0):
            ret.append(''.join(buffer))
            buffer.clear()
        elif(line[ptr] == '"'):
            in_quot += 1
        else:
            buffer.append(line[ptr])
        ptr += 1
    #flush
    if(len(buffer) > 0):
        ret.append(''.join(buffer))
    return ret

def fill_data(cabins, matrix):
    cabin_m1 = [0,0,0]
    cabin_m0 = [0,0,0]

    for i in range(matrix.shape[0]):
        if(matrix[i][FARE] == 0):
            continue
        assert(cabins[i] >= 1 and cabins[i] <= 3)
        cabin_m1[cabins[i]-1] += matrix[i][FARE]
        cabin_m0[cabins[i]-1] += 1
        
    for i in range(3):
        cabin_m1[i] /= cabin_m0[i]

    to_pick = []
    for i in range(matrix.shape[0]):
        if(matrix[i][FARE] == 0):
            assert(cabins[i] >= 1 and cabins[i] <= 3)
            matrix[i][FARE] = cabin_m1[cabins[i]-1]
        if(matrix[i][AGE] != 0):
            to_pick.append(i)

    NUM_NEAREST_NEIGHBORS = 10
    nn = NearestNeighbors(n_neighbors=NUM_NEAREST_NEIGHBORS, algorithm='ball_tree')
    special = matrix[to_pick, :]
    nn.fit(special)
    for i in range(matrix.shape[0]):
        if(matrix[i][3] == -1):
            indices = nn.kneighbors(matrix[i].reshape(1, -1), return_distance=False)
            age_avg = 0
            for j in range(indices.shape[1]):
                   age_avg += matrix[indices[0][j]][AGE]
            age_avg /= 10
            matrix[i][AGE] = age_avg

def preprocess():
    #read in the file
    infile = open('train.csv', 'r')
    matrix = np.ndarray(shape=(NUM_LINES, 5), dtype=float)
    cabins = np.ndarray(shape=(NUM_LINES,), dtype=int)
    labels = np.ndarray(shape=(NUM_LINES,), dtype=int)

    ctr = -1
    for line in infile:
        if(ctr == -1):
            ctr = 0
            continue

        str_array = split_comma(line)
        survived = 0
        cabin = 0
        gender = 0
        age = 0
        sibsp = 0
        parch = 0
        fare = 0

        if(str_array[1] == '0'):
            survived = -1
        elif(str_array[1] == '1'):
            survived = 1
        else:
            continue
        cabin = 2 if(str_array[2] == '') else float(str_array[2])
        if(str_array[4] == 'male'):
            gender = 1
        elif(str_array[4] == 'female'):
            gender = -1
        else:
            gender = 0
        if(str_array[5] == ''):
            age = -1
        else:
            age = float(str_array[5])
        if(str_array[6] == ''):
            sibsp = 0
        else:
            sibsp = float(str_array[6])
        if(str_array[7] == ''):
            parch = 0
        else:
            parch = float(str_array[7])
        if(str_array[9] == ''):
            fare = -1
        else:
            fare = float(str_array[9])

        cabins[ctr] = cabin
        matrix[ctr][FARE] = fare
        matrix[ctr][FAMILY] = sibsp+parch
        matrix[ctr][AGE] = age
        matrix[ctr][GENDER] = gender
        labels[ctr] = survived
        ctr += 1

    print(cabins)
    fill_data(cabins, matrix)
    print(cabins)
    return matrix,labels

def read_tests():
    infile = open('test.csv', 'r')
    matrix = np.ndarray(shape=(NUM_LINES, 4), dtype=float)
    cabins = np.ndarray(shape=(NUM_LINES,), dtype=int)

    ctr = -1
    for line in infile:
        if(ctr == -1):
            ctr = 0
            continue

        str_array = split_comma(line)
        cabin = 0
        gender = 0
        age = 0
        sibsp = 0
        parch = 0
        fare = 0

        cabin = 2 if(str_array[1] == '') else float(str_array[1])
        if(str_array[3] == 'male'):
            gender = 1
        elif(str_array[3] == 'female'):
            gender = -1
        else:
            gender = 0
        if(str_array[4] == ''):
            age = -1
        else:
            age = float(str_array[4])
        if(str_array[5] == ''):
            sibsp = 0
        else:
            sibsp = float(str_array[5])
        if(str_array[6] == ''):
            parch = 0
        else:
            parch = float(str_array[6])
        if(str_array[8] == ''):
            fare = -1
        else:
            fare = float(str_array[8])

        cabins[ctr] = cabin
        matrix[ctr][FARE] = fare
        matrix[ctr][FAMILY] = sibsp+parch
        matrix[ctr][AGE] = age
        matrix[ctr][GENDER] = gender
        ctr += 1
        
    fill_data(cabins, matrix)
    return matrix

def extract(matrix, col, train_labels):
    columns = dict()
    pos_ld = dict()
    neg_ld = dict()
    column = []
    sizes = []
    labels = []
    for i in range(matrix.shape[0]):
        rnd = round(matrix[i][col],3)
        if(rnd in columns):
            columns[rnd] += 1
        else:
            columns[rnd] = 1
        if(train_labels[i] == 1):
            if(rnd in pos_ld):
                pos_ld[rnd] += train_labels[i]
            else:
                pos_ld[rnd] = train_labels[i]
        else:
            if(rnd in neg_ld):
                neg_ld[rnd] -= train_labels[i]
            else:
                neg_ld[rnd] = -train_labels[i]
    for k,v in columns.items():
        column.append(k)
        column.append(k)
        if(k in pos_ld):
            sizes.append(1+pos_ld[k]//2)
        else:
            sizes.append(0)
        if(k in neg_ld):
            sizes.append(1+neg_ld[k]//2)
        else:
            sizes.append(0)
        labels.append(1)
        labels.append(-1)
    return column,sizes,labels

def run():
    train_matrix, train_labels = preprocess()
    np.savetxt('corrected.csv', train_matrix, delimiter=',')
    test_matrix = read_tests()

    for j in range(5):
        col, siz, labels = extract(train_matrix, j, train_labels)
        plt.scatter(col, labels, s=siz)
        plt.show()
    
