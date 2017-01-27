import numpy as np
import pickle 
from math import sqrt
import timeit
from datetime import datetime
import os
import zipfile



'''
dim = 20                    #dimension of embedding space of entities and relations
shuffle_data = True         #shuffle training data before each epoch
check_collision = True      #when creating corrupt training triples, check if triples is truly non-existent in whole data set 
dataset = 'Freebase'           #Alternatively 'Wordnet', can be extended to any dataset, please make sure to set 'swap' accordingly 
device = 'cpu' 		    # alternatively: gpu
swap = True                 #swap colums of raw triple files to bring to format 'subject predicate object', if not already in this format
max_epoch = 4000             #maximal epoch number for training 
#global_epoch = 0            #number of epochs model was trained, global across multiple training sessions of the same model
margin = 2                  #param for loss function; greater than 0, e.g. {1, 2, 10}
learning_rate = 0.1         #for Adagrad optimizer
batch_size = 100            #mini-batch-size in every training iteration
l1_flag = True    	    #for dissimilarity measure during training
result_log_cycle = 250        #run validation and log results after every x epochs
#embedding_log_cycle = 5     #save embedding to disk after every x epochs
test_size = 25000            #to speed up validation validate on a randomly drawn set of x from validation set
valid_verbose = False       #display scores for each test triple during validation 
train_verbose = False       #display loss after each gradient step during training
normalize_ent = True  	    #normalization of entity vectors after each gradient step 
'''
class KGC_Model:
    def __init__(self, dataset, swap, model_name, dim, shuffle_data=True, check_collision=True):
        self.dataset = dataset
        self.swap = swap
        self.model_name = model_name
        self.dim = dim
        self.shuffle_data = shuffle_data
        self.check_collision = check_collision

    def getPATHS(self):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
        PATH = '../../../data/Trained Models for test/'+ self.model_name+'/' + self.dataset + '/dim = '+str(self.dim) +'/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        PLOT_PATH = '../../../data/Model Validation Results for Plotting for test/' + self.dataset + '/dim = '+str(self.dim) +'/'
        if not os.path.exists(PLOT_PATH):
            os.makedirs(PLOT_PATH)

        MODEL_META_PATH = PATH + self.model_name + '_model_meta.txt'
        INITIAL_MODEL = PATH + self.model_name + '_initial_model'
        MODEL_PATH = PATH + self.model_name + '_model'
        RESULTS_PATH = PATH + self.model_name + '_results'

        PLOT_RESULTS_PATH = PLOT_PATH + self.model_name + '_results'
        PLOT_MODEL_META_PATH = PLOT_PATH + self.model_name + '_model_meta.txt'
        return [PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH], [PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH]


    #helper methods creation of input files  

    #creates int-to-URI dicts for entities and relations 
    #input is the complete triple-store 

    def parse_triple_store(self, triples):  
        #else create new dicts
        e1 = triples[:,0]   #subject
        e2 = triples[:,2]   #object 
        rel = triples[:,1]  #predicate
        ent_int_to_URI = list(set(e1).union(set(e2)))
        rel_int_to_URI = list(set(rel)) 
        return ent_int_to_URI, rel_int_to_URI

    #only URI-to-int dicts are saved to disk
    #int_to_URI dicts are needed to create the former
    def create_URI_to_int(self, ent_int_to_URI, rel_int_to_URI):
        ent_URI_to_int = {ent_int_to_URI[i]: i for i in range(len(ent_int_to_URI))}    
        rel_URI_to_int = {rel_int_to_URI[i]: i for i in range(len(rel_int_to_URI))}
        return ent_URI_to_int, rel_URI_to_int

    #method for preprocessing triple files such that columns are swapped to format sub-pred-obj
    def swap_cols(self, arr, frm, to):
        arr[:,[frm, to]] = arr[:,[to, frm]] 

    #create integer matrix from triple-store usings dicts
    def create_matrix(self, triples, ent_URI_to_int, rel_URI_to_int): 
        int_matrix =  np.asarray([[ent_URI_to_int[triples[i,0]], rel_URI_to_int[triples[i,1]], ent_URI_to_int[triples[i,2]]] for i in range(len(triples))], dtype=np.int32)
        return int_matrix


    def create_triple_array_batches(self, matrix_batch, ent_array_map, rel_array_map, corrupt=False):
        h_batch =  np.asarray([ent_array_map[matrix_batch[i,0]] for i in range(len(matrix_batch))])
        t_batch =  np.asarray([ent_array_map[matrix_batch[i,2]] for i in range(len(matrix_batch))])

        if corrupt: 
            return h_batch, t_batch
        l_batch =  np.asarray([rel_array_map[matrix_batch[i,1]] for i in range(len(matrix_batch))])
        return h_batch, l_batch, t_batch


    def corrupt_triple_matrix(self, triples_set, matrix_batch, n, check_collision=True): 
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
    def load_data(self):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
        DATA_PATH = '../../../data/Triple Store/' + self.dataset + '/'
        #if any of the triple files is missing unpack data.zip
        if not os.path.isfile(DATA_PATH +'train.txt') or not os.path.isfile(DATA_PATH +'test.txt') or not os.path.isfile(DATA_PATH +'valid.txt'): 
            zip_ref = zipfile.ZipFile(DATA_PATH +'data.zip', 'r')
            zip_ref.extractall(DATA_PATH)
            zip_ref.close()

        train_triples = np.loadtxt(DATA_PATH +'train.txt',dtype=np.object,comments='#',delimiter=None)
        test_triples = np.loadtxt(DATA_PATH +'test.txt',dtype=np.object,comments='#',delimiter=None) #59071 triples
        valid_triples = np.loadtxt(DATA_PATH +'valid.txt',dtype=np.object,comments='#',delimiter=None) #50000 triples

        if self.swap: 
            self.swap_cols(train_triples, 1,2)
            self.swap_cols(test_triples, 1,2)
            self.swap_cols(valid_triples, 1,2)
        triples = np.concatenate((train_triples, test_triples, valid_triples), axis=0)
        return triples, train_triples, valid_triples, test_triples


    #create URI_to_int dicts for entitiy and relations 
    def create_dicts(self, triples):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
        DATA_PATH = '../../../data/Triple Store/' + self.dataset + '/'
        if os.path.isfile(DATA_PATH +'URI_to_int'): 
            URI_to_int = self.pickle_object(DATA_PATH +'URI_to_int', 'r')
            ent_URI_to_int = URI_to_int[0]
            rel_URI_to_int = URI_to_int[1]
            return ent_URI_to_int, rel_URI_to_int
        else: 
            ent_int_to_URI, rel_int_to_URI = self.parse_triple_store(triples)
            ent_URI_to_int, rel_URI_to_int = self.create_URI_to_int(ent_int_to_URI, rel_int_to_URI)
            URI_to_int = [ent_URI_to_int, rel_URI_to_int]
            self.pickle_object(DATA_PATH + 'URI_to_int', 'w', URI_to_int)
            return ent_URI_to_int, rel_URI_to_int

    #given URI triple matrices and URI-to-int dicts create int triple matrices and
    #set of triples for faster search through triple store 
    def create_int_matrices(self, triples, train, valid, test, ent_URI_to_int, rel_URI_to_int):
        triples_matrix = self.create_matrix(triples, ent_URI_to_int, rel_URI_to_int)
        train_matrix = self.create_matrix(train, ent_URI_to_int, rel_URI_to_int)
        valid_matrix = self.create_matrix(valid, ent_URI_to_int, rel_URI_to_int)
        test_matrix = self.create_matrix(test, ent_URI_to_int, rel_URI_to_int)

        triples_set = set((tuple(triples_matrix[i])) for i in range(len(triples_matrix)))
        return triples_set, train_matrix, valid_matrix, test_matrix 

    #method for either reading from or writing to disk a pickle object
    def pickle_object(self, FILE_NAME, mode, object_=None):
        if mode == 'w':
            file = open(FILE_NAME, 'w')
            pickle.dump(object_, file)
            file.close()
        if mode == 'r':
            file = open(FILE_NAME, 'r')
            object_ = pickle.load(file)
            file.close()
            return object_   
