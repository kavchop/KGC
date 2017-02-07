import numpy as np
import pickle 
from math import sqrt
import timeit
from datetime import datetime
import os
import zipfile


###################

#helper methods creation of input files  

#creates int-to-URI dicts for entities and relations 
#input is the complete triple-store 
def parse_triple_store(triples):  
    #else create new dicts
    e1 = triples[:,0]   #subject
    e2 = triples[:,2]   #object 
    rel = triples[:,1]  #predicate
    ent_int_to_URI = list(set(e1).union(set(e2)))
    rel_int_to_URI = list(set(rel)) 
    return ent_int_to_URI, rel_int_to_URI

#only URI-to-int dicts are saved to disk
#int_to_URI dicts are needed to create the former
def create_URI_to_int(ent_int_to_URI, rel_int_to_URI):
    ent_URI_to_int = {ent_int_to_URI[i]: i for i in range(len(ent_int_to_URI))}    
    rel_URI_to_int = {rel_int_to_URI[i]: i for i in range(len(rel_int_to_URI))}
    return ent_URI_to_int, rel_URI_to_int

#method for preprocessing triple files such that columns are swapped to format sub-pred-obj
def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]] 

#create integer matrix from triple-store usings dicts
def create_matrix(triples, ent_URI_to_int, rel_URI_to_int): 
    int_matrix =  np.asarray([[ent_URI_to_int[triples[i,0]], rel_URI_to_int[triples[i,1]], ent_URI_to_int[triples[i,2]]] for i in range(len(triples))], dtype=np.int32)
    return int_matrix



def create_triple_array_batches(matrix_batch, ent_array_map, rel_array_map, corrupt=False):
    h_batch =  np.asarray([ent_array_map[matrix_batch[i,0]] for i in range(len(matrix_batch))])
    t_batch =  np.asarray([ent_array_map[matrix_batch[i,2]] for i in range(len(matrix_batch))])

    if corrupt: 
        return h_batch, t_batch
    l_batch =  np.asarray([rel_array_map[matrix_batch[i,1]] for i in range(len(matrix_batch))])
    return h_batch, l_batch, t_batch


def corrupt_triple_matrix(triples_set, matrix_batch, n, check_collision=True): 
    corrupt_batch = np.copy(matrix_batch) 
    if check_collision: 
        for i in range(len(corrupt_batch)):
            while tuple(corrupt_batch[i]) in triples_set:
                a = np.random.randint(n)  #n is number of all unique entities 
                cor_ind = np.random.choice((0,2))  #randomly select index to corrupt (head or tail)
                corrupt_batch[i][cor_ind]= a
    else: 
        for i in range(len(corrupt_batch)):
            a = np.random.randint(n)  #n is number of all unique entities 
            cor_ind = np.random.choice((0,2))
            corrupt_batch[i][cor_ind]= a
    return corrupt_batch



#load train, test and validation file, preprocess (swap columns if need be) and output
#all triple stores as numpy matrices of URIs
def load_data(dataset, swap=False):
    #set working directory to current directory and based on this set all required PATHS
    os.chdir(os.getcwd())
    DATA_PATH = '../../../data/Triple Store/' + dataset + '/'
    #if any of the triple files is missing unpack data.zip
    if not os.path.isfile(DATA_PATH +'train.txt') or not os.path.isfile(DATA_PATH +'test.txt') or not os.path.isfile(DATA_PATH +'valid.txt'): 
        zip_ref = zipfile.ZipFile(DATA_PATH +'data.zip', 'r')
        zip_ref.extractall(DATA_PATH)
        zip_ref.close()
    
    train_triples = np.loadtxt(DATA_PATH +'train.txt',dtype=np.object,comments='#',delimiter=None)
    test_triples = np.loadtxt(DATA_PATH +'test.txt',dtype=np.object,comments='#',delimiter=None) #59071 triples
    valid_triples = np.loadtxt(DATA_PATH +'valid.txt',dtype=np.object,comments='#',delimiter=None) #50000 triples
    
    if swap: 
        swap_cols(train_triples, 1,2)
        swap_cols(test_triples, 1,2)
        swap_cols(valid_triples, 1,2)
    triples = np.concatenate((train_triples, test_triples, valid_triples), axis=0)
    return triples, train_triples, valid_triples, test_triples


#create URI_to_int dicts for entitiy and relations 
def create_dicts(dataset, triples):
    #set working directory to current directory and based on this set all required PATHS
    os.chdir(os.getcwd())
    DATA_PATH = '../../../data/Triple Store/' + dataset + '/'
    if os.path.isfile(DATA_PATH +'URI_to_int'): 
        URI_to_int = pickle_object(DATA_PATH +'URI_to_int', 'r')
        ent_URI_to_int = URI_to_int[0]
        rel_URI_to_int = URI_to_int[1]
        return ent_URI_to_int, rel_URI_to_int
    else: 
        ent_int_to_URI, rel_int_to_URI = parse_triple_store(triples)
        ent_URI_to_int, rel_URI_to_int = create_URI_to_int(ent_int_to_URI, rel_int_to_URI)
        URI_to_int = [ent_URI_to_int, rel_URI_to_int]
        pickle_object(DATA_PATH + 'URI_to_int', 'w', URI_to_int)
        return ent_URI_to_int, rel_URI_to_int

#given URI triple matrices and URI-to-int dicts create int triple matrices and
#set of triples for faster search through triple store 
def create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int):
    triples_matrix = create_matrix(triples, ent_URI_to_int, rel_URI_to_int)
    train_matrix = create_matrix(train, ent_URI_to_int, rel_URI_to_int)
    valid_matrix = create_matrix(valid, ent_URI_to_int, rel_URI_to_int)
    test_matrix = create_matrix(test, ent_URI_to_int, rel_URI_to_int)
    
    triples_set = set((tuple(triples_matrix[i])) for i in range(len(triples_matrix)))
    return triples_set, train_matrix, valid_matrix, test_matrix 

#method for either reading from or writing to disk a pickle object
def pickle_object(FILE_NAME, mode, object_=None):
    if mode == 'w':
        file = open(FILE_NAME, 'w')
        pickle.dump(object_, file)
        file.close()
    if mode == 'r':
        file = open(FILE_NAME, 'r')
        object_ = pickle.load(file)
        file.close()
        return object_
        
def numpy_norm(x, l1_flag=True):
    if l1_flag:
        return np.linalg.norm((x), ord=1)
    else: 
        return np.linalg.norm(x)

def normalize_embedding(x, l1_flag=True):
    l1_flag=False	
    return np.array([x[i]/numpy_norm(x[i], l1_flag) for i in range(len(x))])

def normalize_vec(x, l1_flag=True):
    l1_flag=False
    return x/numpy_norm(x, l1_flag)

#computes the distance score of a triple batch: x=(h+l-t)
#used in validation part 
def score_func(h, l, t, l1_flag=True):
    return -1 * numpy_norm(h+l-t, l1_flag)

def init_params(n,m, dim):
    l1_flag = False
    ent_array_map = np.random.uniform(-6/sqrt(dim),6/sqrt(dim), (n, dim))
    rel_array_map = np.random.uniform(-6/sqrt(dim),6/sqrt(dim), (m, dim))

    ent_array_map = normalize_embedding(ent_array_map, l1_flag)
    rel_array_map = normalize_embedding(rel_array_map, l1_flag)
    return ent_array_map, rel_array_map

def getPATHS(model_name, dim, dataset):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
	PATH = '../../../data/Trained Models/'+model_name+'/' + dataset + '/dim = '+str(dim) +'/'
	if not os.path.exists(PATH):
	    os.makedirs(PATH)

	PLOT_PATH = '../../../data/Model Validation Results for Plotting/' + dataset + '/dim = '+str(dim) +'/'
	if not os.path.exists(PLOT_PATH):
	    os.makedirs(PLOT_PATH)

	MODEL_META_PATH = PATH + model_name + '_model_meta.txt'
	INITIAL_MODEL = PATH + model_name + '_initial_model'
	MODEL_PATH = PATH + model_name + '_model'
	RESULTS_PATH = PATH + model_name + '_results'

	PLOT_RESULTS_PATH = PLOT_PATH + model_name + '_results'
	PLOT_MODEL_META_PATH = PLOT_PATH + model_name + '_model_meta.txt'
        return [PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH], [PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH]

#method writes model configurations to disk 
def save_model_meta(model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, dim, learning_rate, normalize_ent, check_collision, global_epoch=None, resumed=False):
    if resumed==False: 
	    text_file = open(MODEL_META_PATH, "w")
	    text_file.write("\nmodel: {}\n\n".format(model_name))

	    text_file.write("created on: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
	    text_file.write("embedding dimension:  {}\n".format(dim))
	    text_file.write("learning rate:  {}\n".format(learning_rate))
	    text_file.write("normalized entity vectors:  {}\n".format(normalize_ent))
	    text_file.write("collision check:  {}\n".format(check_collision))
	    text_file.close()

    if resumed==True: 
	    new_lines = "\ntraining resumed on {}\nat epoch: {}\nwith learning rate: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), global_epoch, learning_rate)
	    with open(MODEL_META_PATH, "a") as f:
	    	f.write(new_lines)
