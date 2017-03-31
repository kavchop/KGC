import numpy as np
import pickle 
from math import sqrt
import timeit
from datetime import datetime
import os
import zipfile

# Data Path is provided as global Parameter
os.chdir(os.getcwd())
DATA_PATH = '../../data/'

class KBC_Util_Class:
    def __init__(self, dataset, swap, model_name=None, dim=None):
        self.dataset = dataset
        self.swap = swap
        self.model_name = model_name
        self.dim = dim
	self.DATA_PATH = DATA_PATH
	self.DATASET_PATH = self.DATA_PATH + 'Triple Store/' + self.dataset + '/'
	if model_name != None and dim != 0: 
		self.PATH, self.MODEL_PATH, self.INITIAL_MODEL = self.get_PATHS()
	else: 
		self.MODEL_PATH = None
		
    def get_PATHS(self):
        #set working directory to current directory and based on this set all required PATHS
	os.chdir(os.getcwd())
	#PATH = self.DATA_PATH + 'Trained Models Test/'+ self.model_name + '/' + self.dataset + '/dim='+ str(self.dim) +'/'
	PATH = self.DATA_PATH + 'Trained Models/'+ self.model_name+'/' + self.dataset + '/dim = '+str(self.dim) +'/'
	MODEL_PATH = PATH + self.model_name + '_model'
	INITIAL_MODEL = PATH + self.model_name + '_initial_model'
	return PATH, MODEL_PATH, INITIAL_MODEL 

    def data_exists(self): 
	if not os.path.isfile(self.DATASET_PATH + self.dataset +'.zip'): 
	    print "\nThe specified dataset '{}' does not exist. Please add the dataset (Training, Validation and Test set) in the specified dataset directory before training with it.\n".format(self.dataset)	
	    return False
	if self.MODEL_PATH != None:
	    if os.path.isfile(self.MODEL_PATH):
		return True
	    else:
		print "No model with the specified configurations trained yet!\n"
		return False
	else: 
	    return True


    # load intial embedding maps for entities and relations
    def load_model(self, MODEL_PATH): 
    	model = self.pickle_object(MODEL_PATH, 'r')
	#entity_embed = model[0]
	#relation_embed = model[1]
    	return model


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


    # load URI_to_int dicts for entities and relations 
    def get_URI_to_int(self):
    	file = open(self.DATASET_PATH+'URI_to_int', 'r') 
    	URI_to_int = pickle.load(file)
    	ent_URI_to_int = URI_to_int[0]
    	rel_URI_to_int = URI_to_int[1]
    	file.close()
    	return ent_URI_to_int, rel_URI_to_int 


    # load URI_to_int dicts for entities and relations 
    def get_int_to_URI(self):
    	file = open(self.DATASET_PATH+'URI_to_int', 'r') 
    	URI_to_int = pickle.load(file)
    	ent_URI_to_int = URI_to_int[0]
    	rel_URI_to_int = URI_to_int[1]
    	file.close()
        
        entity_list = {}
        relation_list = {}
        for uri, int_id in ent_URI_to_int.iteritems(): 
	    entity_list[int_id] = uri

        for uri, int_id in rel_URI_to_int.iteritems(): 
	    relation_list[int_id] = uri
    	return entity_list, relation_list 


    #method for preprocessing triple files such that columns are swapped to format sub-pred-obj
    def swap_cols(self, arr, frm, to):
        arr[:,[frm, to]] = arr[:,[to, frm]] 

    #create integer matrix from triple-store usings dicts
    def create_matrix(self, triples, ent_URI_to_int, rel_URI_to_int): 
        int_matrix =  np.asarray([[ent_URI_to_int[triples[i,0]], rel_URI_to_int[triples[i,1]], ent_URI_to_int[triples[i,2]]] for i in range(len(triples))], dtype=np.int32)
        return int_matrix

    def get_triple_matrix(self): 
	triples, train, valid, test = self.load_data()
    	ent_URI_to_int, rel_URI_to_int = self.create_dicts(triples)   
    	triples_set, train_matrix, valid_matrix, test_matrix = self.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)
    	triples = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
	return triples


    def corrupt_triple_matrix(self, triples_set, matrix_batch, n): 
        corrupt_batch = np.copy(matrix_batch) 
        for i in range(len(corrupt_batch)):
        	while tuple(corrupt_batch[i]) in triples_set:
                    a = np.random.randint(n)  #n is number of all unique entities 
                    cor_ind = np.random.choice((0,2))  #randomly select index to corrupt (head or tail)
                    corrupt_batch[i][cor_ind]= a
        return corrupt_batch


    # This methods corrupts a positive triple set, such that either head or tail is replaced by a random entity from the set of entities
    # found in domain or range respectively: e.g. Obama presidentOf Germany as a negative triple  
    def corrupt_triple_matrix_common_rel(self, triples_set, triples_matrix, matrix_batch, x): 
    	corrupt_batch = np.copy(matrix_batch) 

    	h_set = triples_matrix[np.where(matrix_batch[:,1] == x)][:,0]
    	t_set = triples_matrix[np.where(matrix_batch[:,1] == x)][:,1]
 
    	
        for i in range(len(corrupt_batch)):
            	while tuple(corrupt_batch[i]) in triples_set:
                	cor_ind = np.random.choice((0,2))  #randomly select index to corrupt (head or tail)
                	if cor_ind == 0: #head
                		corrupt_batch[i][cor_ind]= np.random.choice((h_set))
			else: 
				corrupt_batch[i][cor_ind]= np.random.choice((t_set))
    	return corrupt_batch


    #load train, test and validation file, preprocess (swap columns if need be) and output
    #all triple stores as numpy matrices of URIs
    def load_data(self):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
        #if any of the triple files is missing unpack data.zip
        if not os.path.isfile(self.DATASET_PATH +'train.txt') or not os.path.isfile(self.DATASET_PATH +'test.txt') or not os.path.isfile(self.DATASET_PATH +'valid.txt'): 
            zip_ref = zipfile.ZipFile(self.DATASET_PATH + self.dataset +'.zip', 'r')
            zip_ref.extractall(self.DATASET_PATH)
            zip_ref.close()

        train_triples = np.loadtxt(self.DATASET_PATH +'train.txt',dtype=np.object,comments='#',delimiter=None)
        test_triples = np.loadtxt(self.DATASET_PATH +'test.txt',dtype=np.object,comments='#',delimiter=None) #59071 triples
        valid_triples = np.loadtxt(self.DATASET_PATH +'valid.txt',dtype=np.object,comments='#',delimiter=None) #50000 triples

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
        if os.path.isfile(self.DATASET_PATH +'URI_to_int'): 
            URI_to_int = self.pickle_object(self.DATASET_PATH +'URI_to_int', 'r')
            ent_URI_to_int = URI_to_int[0]
            rel_URI_to_int = URI_to_int[1]
            return ent_URI_to_int, rel_URI_to_int
        else: 
            ent_int_to_URI, rel_int_to_URI = self.parse_triple_store(triples)
            ent_URI_to_int, rel_URI_to_int = self.create_URI_to_int(ent_int_to_URI, rel_int_to_URI)
            URI_to_int = [ent_URI_to_int, rel_URI_to_int]
            self.pickle_object(self.DATASET_PATH + 'URI_to_int', 'w', URI_to_int)
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
