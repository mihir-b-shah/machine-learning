
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

NUM_LINES = 891
TEST_LINES = 418
    
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

def fare_map(x):
    return int(np.log2(x-3))

def age_map(x):
    return x//10

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

    for i in range(special.shape[0]):
        for j in range(special.shape[1]):
            assert(special[i][j] >= -1000000 and special[i][j] <= 1000000)
    
    for i in range(matrix.shape[0]):
        if(matrix[i][AGE] == -1):
            indices = nn.kneighbors(matrix[i].reshape(1, -1), return_distance=False)
            age_avg = 0
            for j in range(indices.shape[1]):
                   age_avg += matrix[indices[0][j]][AGE]
            age_avg /= 10
            matrix[i][AGE] = age_avg

    # discretize the range and age values
    for i in range(matrix.shape[0]):
        matrix[i][AGE] = age_map(matrix[i][AGE])
        matrix[i][FARE] = fare_map(matrix[i][FARE])

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

    matrix = matrix[:ctr]
    labels = labels[:ctr]
    cabins = cabins[:ctr]
    fill_data(cabins, matrix)
    
    return matrix,labels

def read_tests():
    infile = open('test.csv', 'r')
    matrix = np.ndarray(shape=(TEST_LINES, 4), dtype=float)
    cabins = np.ndarray(shape=(TEST_LINES,), dtype=int)

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

def run():
    train_matrix, train_labels = preprocess()
    test_matrix = read_tests()

    
