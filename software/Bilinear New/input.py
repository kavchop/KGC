import numpy as np
import tensorflow as tf
from datetime import datetime
import timeit 
import pickle 
from bilinear import model_name
import os
import zipfile 

os.chdir(os.getcwd())
DATA_PATH = "../../data/"

    
MODEL_PATH='models/'+model_name +'_model'
INITIAL_MODEL='models/'+model_name+'_initial_model'
RESULTS_PATH='models/'+model_name +'_results'



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
    int_matrix =  np.asarray([[ent_URI_to_int[triples[i,0]], rel_URI_to_int[triples[i,1]], ent_URI_to_int[triples[i,2]]] for i in range(len(triples))])
    return int_matrix

#create vector files using int matrix and entity-to-vector map
def create_triple_array_batches(matrix_batch, ent_array_map):
    h_batch =  np.asarray([ent_array_map[matrix_batch[i,0]] for i in range(len(matrix_batch))])
    t_batch =  np.asarray([ent_array_map[matrix_batch[i,2]] for i in range(len(matrix_batch))])
    return h_batch, t_batch


#create map of triple batches by relation 
#during training iteration over such map to process the corresponding Mr for every mini-batch 
def create_triples_map_by_rel(int_matrix, m):
    triples_map = [np.reshape(np.array([], dtype=np.int32), (0,3)) for i in range(m)]

    for i in range(len(int_matrix)): 
        new_record =  np.reshape(int_matrix[i], (1,3))
        triples_map[int_matrix[i,1]]= np.append(triples_map[int_matrix[i,1]], new_record, axis=0)  
    return triples_map


# creates corrupted int matrix given positive training matrix 
def create_corrupt_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision): 
    if corrupt_two: 
        corrupt_batch_head = corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision, corrupt_head=True)
        corrupt_batch_tail = corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision, corrupt_head=False)
        corrupt_batch = np.append(corrupt_batch_head , corrupt_batch_tail , axis=0)
        return corrupt_batch
    else: 
        return corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision)


#corrupt randomly head of positive matrix or return matrix where only head or tail is corrupted
def corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision, corrupt_head=None): 
    if corrupt_head:
        cor_ind = 0  #corruption index: head
    else: 
        cor_ind = 2  # tail
        
    corrupt_batch = np.copy(matrix_batch) 
    if check_collision: 
        for i in range(len(corrupt_batch)):
            while tuple(corrupt_batch[i]) in triples_set:
                a = np.random.randint(n)  #n is number of all unique entities 
                if not corrupt_two:       #if random corruption  
                    cor_ind =  np.random.choice((0,2))
                corrupt_batch[i][cor_ind]= a
    else: 
        for i in range(len(corrupt_batch)):
            a = np.random.randint(n)  #n is number of all unique entities 
            if not corrupt_two: 
                cor_ind =  np.random.choice((0,2))
            corrupt_batch[i][cor_ind]= a
    return corrupt_batch

#load train, test and validation file, preprocess (swap columns if need be) and output
#all triple stores as numpy matrices of URIs
def load_data(swap=False):
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
def create_dicts(triples):
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
        
#normalize n x n_red matrix row-wise
def normalize_W(W):
    return np.array([W[i]/np.linalg.norm(W[i]) for i in range(len(W))])

#initialize model parameters W and Mr
def init_params(m, n, n_red, a):  
    W = np.random.rand(n,n_red)
    A = np.random.rand(m, n_red, a)
    B = np.random.rand(m, n_red, a)
    #normalize entitiy matrix
    W = normalize_W(W)
    return W, A, B

# diagonal matrix as 1 hot map for initial entity embedding of dim=n 
def create_1_hot_maps(n): 
    return np.eye(n)

#create matrix for learned entity embedding of dim=n_red: y = x*W where x is 1-hot vector 
def learned_ent_embed(ent_array_map, W_param):
    entity_embed = np.array([np.dot(ent_array_map[i], W_param) for i in range(len(ent_array_map))])
    return entity_embed

#method writes model configurations to disk 
def save_model_meta(model_name, n_red, learning_rate, corrupt_two, normalize_ent, check_collision):
    text_file = open("models/"+model_name+"_model_meta.txt", "w")
    text_file.write("\nmodel: {}\n\n".format(model_name))

    text_file.write("created on: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
    text_file.write("factorization rank (dim.):  {}\n".format(n_red))
    text_file.write("learning rate:  {}\n".format(learning_rate))
    text_file.write("two corrupted per positive:  {}\n".format(corrupt_two))
    text_file.write("normalized entity matrix:  {}\n".format(normalize_ent))
    text_file.write("collision check:  {}\n".format(check_collision))
    text_file.close()
