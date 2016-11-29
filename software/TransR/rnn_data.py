'''
Author of this implementation: Kavita Chopra (2016, version 1.0)

Source of conceptual model:

Bilinear model using Alternative Least Square Optimization (RESCAL): 
- "A three way model for collective learning on multi-relational data", 
   Nickel et al., 2011
  "Factorization of Yago" - Nickel et al., 2012
Tensor Network Formulation of Rescal: 
-  "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", 
    B. Yang et al. (2014)

Description of the Representation Learning Model
- Rescal is based on the factorization of a 3 mode tensor (3 dimensional matrix)
  where the aim is to solve the model parameters A and Mr given the low rank factorization 
  Xk = A x Mr x A^T
- A is a n x n_red (with rank n_red << n (#entities)) matrix representing the entity embedding
- Mr is a r x n_red matrix and there are as many different Mr's as there are relations in the triple store

The Neural Tensor Network Approach
- In a neural tensor network approach, this factorization is solved through a 2 layer network: 
- 1st layer: yi = xi * W  where xi is a one-hot vector for an entity representation (n-dimensional)
                      and W is a n x r matrix parameter that is to be learned
- 2nd layer: score = yh * Mr * yt where yh is the learned entity representation for a head
                                  Mr is the relation matrix for relation r and yt is the 
                                  learned entity representation for the tail entitiy 
- in the 2nd layer scores are computed for the positive and negative triple and optimized 
  by the loss function in a way, that positive triples get a higher score and negative triples a lower 


Variation of Rescal: Bilinear Diagonal Model
- In the bilinear diagonal model Mr is restricted to be a diagonal matrix to reduce the 
  complexity of the model 
- see "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", 
  B. Yang et al. (2014)
- run diagonal-bilinear training with "python bilinear.py --diagonal"
                                  


Input
- training data consisting of two sets: positive training data (true triples), and 
- negative training data, where for every positive triple there are two corrupted
  triples: one where head is replaced and the other where tail is corrupted  
- Hyperparameters: 
  factorization rank (dimension to which n is reduced to) n_red, learning rate

Optimization Procedure
- stochastic gradient descent with mini-batches using Adagrad-method

Knowledge Base
- subset of Freebase (FBK15) with frequent entities also present in Wikilinks

Split of Data
- training data: ~480,000 triples
- validation set   50,000    "
- test set         59,000    "
- all sets are disjoint
- validation on valid data set used to measure performance of model during training and 
  for employment of early stopping before a maximimum epoch-bound (e.g. 1000)

Validation and Evaluation Protocol
- for validation mean ranks of correct triples from a list of corrupted triples are reported
- evaluation on test data after training is complete to measure the quality of the final model 
- for evaluation hits at ten (proportion of true triples from top ten ranks) are reported

Implementation Remarks
- run code with command line "python bilinear.py" for training with non-diagonal Mr or 
  "python bilinear.py --diagonal" to train the bilinear diagonal model 
- before training a new model:
    - meta-data-file with the customized configurations is created and saved in 'models/' directory
    - initial embedding is saved to disk for visualization purposes e.g. after dimensionality reduction through PCA
- at customizable intervals current model is saved to disk so that training may be continued in different sessions
- global- and hyper-parameters of the model can be configured in the params.py file. 

'''

import numpy as np
from random import shuffle 
import tensorflow as tf
from datetime import datetime
import timeit 
import pickle 
import os
import zipfile 

os.chdir(os.getcwd())
DATA_PATH = "../../data/"





class RandomWalk:
    def __init__(self, triples, seq_len, train_size, num_entities, different_lengths=False):
        self.triples = triples
        self.seq_len = seq_len
        self.train_size = train_size
        self.num_entities = num_entities
        self.different_lengths = different_lengths

    def generate_seq(self): 
        embedding = np.eye(self.num_entities)
        triple_index_list = np.arange(len(self.triples))

        sequence_list= []
        sequence = []
        target_vector = np.reshape(np.array([]), (0,self.num_entities))
        if self.different_lengths: 
            current_seq_len = np.random.randint(3, self.seq_len)
        else: 
            current_seq_len = self.seq_len
        
        x = np.random.randint(0, len(triple_index_list))
        prev_obj = self.triples[triple_index_list[x],2]   
        sequence = sequence + self.triples[triple_index_list[x],0:1].tolist()  # all three .tolist()   ##########[x,:]
        print "first sequence"
        print sequence
    
        while (len(sequence_list)<self.train_size):
            if len(sequence_list)%5==0:
                np.random.shuffle(triple_index_list)
            if len(sequence_list) == 0 and prev_obj==None:        #if no such entity found, start from a new random triple
                x = np.random.randint(0, len(triple_index_list))  #find a new random object of a triple and break the while loop
                prev_obj = self.triples[triple_triple_index_list[x],2]  
            x = None
            for i in range(len(triple_index_list)):   #iterate over triple list and collect all indices where subject matches prev_obj
                #print i 
                if self.triples[triple_index_list[i],0] == prev_obj: # and len(index_list) < 500:    #if we find a triple where the subject matches the object from last triple
                    x = i
                    break
            if x == None:        #if no such entity found, start from a new random triple
                x = np.random.randint(0, len(triple_index_list))  #find a new random object of a triple and break the while loop
                prev_obj = self.triples[triple_index_list[x],2]  
                sequence = []
                sequence = sequence + self.triples[triple_index_list[x],0:1].tolist()
            
            else: #if candidate entities found, 
                if len(sequence) == current_seq_len-1:
                    sequence = sequence + self.triples[triple_index_list[x],2:3].tolist()  #one less 
                    #print sequence 
                    sequence_list.append(sequence)
                    target_vector = np.append(target_vector, np.reshape(embedding[self.triples[triple_index_list[x],2]], (1, self.num_entities)), axis = 0)
                    if len(sequence_list) % 100==0: 
                        print len(sequence_list), '\n'
                        #print sequence_list
                    if len(sequence_list) % 1000==0: 
                        pickle_object(DATA_PATH +'RNN_train', 'w', np.asarray(sequence_list))
                        pickle_object(DATA_PATH +'RNN_target', 'w', target_vector)
                    sequence = []
                    prev_obj = None
                    if self.different_lengths: 
                        current_seq_len = np.random.randint(3, self.seq_len)
                    else: 
                        current_seq_len = self.seq_len
                else: 
                    sequence = sequence + self.triples[triple_index_list[x],2:3].tolist()  #.tolist()  #####1:3
                    #print sequence
                    prev_obj = self.triples[triple_index_list[x],2]
   
        return sequence_list, target_vector




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
def load_data(swap=None):
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
def init_params(m, n, n_red): 
    if FLAGS.diagonal:
        Mr = np.random.rand(m, n_red,)
    else: 
        Mr = np.random.rand(m, n_red, n_red)
    W = np.random.rand(n,n_red)
    #normalize entitiy matrix
    W = normalize_W(W)
    return W, Mr

# diagonal matrix as 1 hot map for initial entity embedding of dim=n 
def create_1_hot_maps(n): 
    return np.eye(n)

#create matrix for learned entity embedding of dim=n_red: y = x*W where x is 1-hot vector 
def learned_ent_embed(ent_array_map, W_param):
    entity_embed = np.array([np.dot(ent_array_map[i], W_param) for i in range(len(ent_array_map))])
    return entity_embed


def main():
#load set of all triples, train, valid and test data
    swap = False
    triples, train, valid, test = load_data(swap)
    ent_URI_to_int, rel_URI_to_int = create_dicts(triples)   #load dicts 

    
    #load int-matrices and triples-set for faster search
    triples_set, train_matrix, valid_matrix, test_matrix  = create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    
    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations
    '''
    #create 1 hot matrix for initial entity representation  
    ent_array_map = create_1_hot_maps(n)

    #create map of triple batches by relations 
    #during training iteration over such map to process the corresponding Mr for every mini-batch 
    triples_map = create_triples_map_by_rel(train_matrix, m)
    '''
    #myFirstObject = RandomWalk(seq_len, train_size, num_entities, different_lengths=False)
    '''
    triples_matrix = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
    print triples_matrix.shape
    num_entities = n
    #triples = np.random.randint(0, high=num_entities, size=(100, 3))
    start = timeit.default_timer()
    myFirstObject = RandomWalk(triples_matrix, 7, 50000, num_entities, different_lengths=False)
 
    seq_list, target_vec = myFirstObject.generate_seq()
    
    pickle_object(DATA_PATH +'RNN_train_1', 'w', np.asarray(seq_list))
    pickle_object(DATA_PATH +'RNN_target_1', 'w', target_vec)
    stop = timeit.default_timer()
    
    print (stop - start)/60

    #print seq_list
    #print target_vec
    #print np.asarray(seq_list)

    start = timeit.default_timer()
    triples_matrix = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
    x = len(triples_matrix)
    #x = 10
    num_vec = n + m
    sequence_list= []
    embedding = np.eye(num_vec)
    target_vector = np.reshape(np.array([]), (0, num_vec))
    for i in range(x):
        sequence = []
        sequence = sequence + [int(triples_matrix[i,0])]
        sequence = sequence + [int(triples_matrix[i,1])+n]
        sequence_list.append(sequence)
        target_vector = np.append(target_vector, np.reshape(embedding[int(triples_matrix[i,2])], (1, num_vec)), axis = 0)

    print np.asarray(sequence_list)
    print target_vector
    '''
    
    start = timeit.default_timer()
    triples_matrix = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
    train_data = triples_matrix[:,0:2]
    #x = len(triples_matrix)
    x = 100000
    num_vec = n + m
    embedding = np.eye(num_vec)
    target_vector = np.reshape(np.array([]), (0, num_vec))
    for i in range(x):
        train_data[i,1]=train_data[i,1]+n
        target_vector = np.append(target_vector, np.reshape(embedding[int(triples_matrix[i,2])], (1, num_vec)), axis = 0)
        if i% 1000==0:
            print i
            pickle_object(DATA_PATH +'RNN_train_' + i, 'w', train_data[0:i])
            pickle_object(DATA_PATH +'RNN_target_' + i, 'w', target_vector)
            
    #print train_data
    #print target_vector
    
    stop = timeit.default_timer()
    
    print (stop - start)/60
    
if __name__=="__main__": 
    #tf.app.run()
    main()    



