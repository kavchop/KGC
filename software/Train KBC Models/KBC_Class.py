import numpy as np
import tensorflow as tf
import pickle 
from math import sqrt
import timeit
from datetime import datetime
import os
import zipfile

# Class KBC_Data has methods which all score based KBC-models have in common

class KBC_Class:
    def __init__(self, dataset, swap, model_name, dim, margin, device, memory, learning_rate, max_epoch, batch_size, test_size, result_log_cycle, eval_with_np=True, shuffle_data=True, check_collision=True, normalize_ent=True):

	# set working directory to current directory and based on this set all required PATHS
	os.chdir(os.getcwd())
	# location of the data path: 
	self.DATA_PATH = '../../data/'
	# Note that all necessary directories will be created based on this path. It is only important to ensure that sub-directory of DATA_PATH: /Triples Store/ must contain directories with datasets used for training, e.g. Freebase, Wordnet, etc., see method load_data() to see how subset_paths are built

        self.dataset = dataset
        self.swap = swap
        self.model_name = model_name
        self.dim = dim                  # latent embedding dim of entities (relations may have a different embedding dim, or the same) 
	self.margin = margin
	self.device = device
	self.memory = memory 
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
	self.batch_size = batch_size
        self.test_size = test_size
        self.result_log_cycle = result_log_cycle
    	self.eval_with_np = eval_with_np
        self.shuffle_data = shuffle_data
        self.check_collision = check_collision
	self.normalize_ent = normalize_ent
	
    
    def getPATHS(self):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
        PATH = self.DATA_PATH + 'Trained Models/'+ self.model_name+'/' + self.dataset + '/dim = '+str(self.dim) +'/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        PLOT_PATH = self.DATA_PATH + 'Model Validation Results for Plotting/' + self.dataset + '/dim = '+str(self.dim) +'/'
        if not os.path.exists(PLOT_PATH):
            os.makedirs(PLOT_PATH)

        MODEL_META_PATH = PATH + self.model_name + '_model_meta.txt'
        INITIAL_MODEL = PATH + self.model_name + '_initial_model'
        MODEL_PATH = PATH + self.model_name + '_model'
        RESULTS_PATH = PATH + self.model_name + '_results'

        PLOT_RESULTS_PATH = PLOT_PATH + self.model_name + '_results'
        PLOT_MODEL_META_PATH = PLOT_PATH + self.model_name + '_model_meta.txt'
        return [PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH], [PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH]


 
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
        DATA_SET_PATH = self.DATA_PATH + 'Triple Store/' + self.dataset + '/'

        #if any of the triple files is missing unpack data.zip
        if not os.path.isfile(DATA_SET_PATH +'train.txt') or not os.path.isfile(DATA_SET_PATH +'test.txt') or not os.path.isfile(DATA_SET_PATH +'valid.txt'): 
            zip_ref = zipfile.ZipFile(DATA_SET_PATH + self.dataset + '.zip', 'r')
            zip_ref.extractall(DATA_SET_PATH)
            zip_ref.close()

        train_triples = np.loadtxt(DATA_SET_PATH +'train.txt',dtype=np.object,comments='#',delimiter=None)
        test_triples = np.loadtxt(DATA_SET_PATH +'test.txt',dtype=np.object,comments='#',delimiter=None) #59071 triples
        valid_triples = np.loadtxt(DATA_SET_PATH +'valid.txt',dtype=np.object,comments='#',delimiter=None) #50000 triples

        if self.swap: 
            self.swap_cols(train_triples, 1,2)
            self.swap_cols(test_triples, 1,2)
            self.swap_cols(valid_triples, 1,2)
        triples = np.concatenate((train_triples, test_triples, valid_triples), axis=0)
        return triples, train_triples, valid_triples, test_triples


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


    def load_model(self, MODEL_PATH):
        model = self.pickle_object(MODEL_PATH, 'r')
        return model


    def save_model(self, MODEL_PATH, model):
        self.pickle_object(MODEL_PATH, 'w', model)

    #create URI_to_int dicts for entitiy and relations 
    def create_dicts(self, triples):
        #set working directory to current directory and based on this set all required PATHS
        os.chdir(os.getcwd())
        DATA_SET_PATH = self.DATA_PATH + 'Triple Store/' + self.dataset + '/'

        if os.path.isfile(DATA_SET_PATH +'URI_to_int'): 
            URI_to_int = self.pickle_object(DATA_SET_PATH +'URI_to_int', 'r')
            ent_URI_to_int = URI_to_int[0]
            rel_URI_to_int = URI_to_int[1]
            return ent_URI_to_int, rel_URI_to_int
        else: 
            ent_int_to_URI, rel_int_to_URI = self.parse_triple_store(triples)
            ent_URI_to_int, rel_URI_to_int = self.create_URI_to_int(ent_int_to_URI, rel_int_to_URI)
            URI_to_int = [ent_URI_to_int, rel_URI_to_int]
            self.pickle_object(DATA_SET_PATH + 'URI_to_int', 'w', URI_to_int)
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


    def get_eval_tags(self, sys_tags): 
    # default setting for eval_mode, change if arg passed when running code
   	eval_mode = False
	filtered = False
	#sys.argv[0] is transe.py now check for optional tags 
	if len(sys_tags) >= 2:
		if sys_tags[-1] == 'filtered' and sys_tags[-2]=='evaluate':
            		eval_mode = True
	    		filtered = True 
		if sys_tags[-1]=='evaluate':
			eval_mode = True			
	return eval_mode, filtered 


    def model_intro_print(self, train_matrix):
	print "\n****************Training of {} starts!****************\n".format(self.model_name)
        print "Number of Triples in Training data: {}".format(len(train_matrix))
        print "Latent Embedding Dimension: {}".format(self.dim)


    def get_graph_placeholders(self):
	# placeholders for current batch-input (int-arrays) in every gradient step
	# placeholders for true triples 
        h_ph = tf.placeholder(tf.int32, shape=(None))     #head (subject)
        t_ph = tf.placeholder(tf.int32, shape=(None))     #tail (object)

        # ph for common relation
        l_ph = tf.placeholder(tf.int32, shape=(None))     #label (relation)

        # ph for corrupted triple
        h_1_ph = tf.placeholder(tf.int32, shape=(None))      #head from corrupted counterpart triple 
        t_1_ph = tf.placeholder(tf.int32, shape=(None))      #tail from corrupted counterpart triple 
	return h_ph, l_ph, t_ph, h_1_ph, t_1_ph


    def get_loss(self, pos_score, neg_score): 
	return tf.reduce_sum(tf.maximum(tf.sub(neg_score, pos_score)+self.margin, 0))


    # methods which will be overwritten by the inheriting Model-Class 
    # numpy and tensorflow methods for computing the score of the model 
    def tf_score(self, h,l,t):
	return 	
        
    def np_score(self, h,l,t):
	return 

    def eval_and_validate(self, eval_model_=False):
	return


    def update_results_table(self, RESULTS_PATH, PLOT_RESULTS_PATH, triples_set, valid_matrix, model, global_epoch, loss_sum, results_table=None, init=False): 
	if init:
        	results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits', 'total_loss']), (1,6))
		# note that only running validation (eval_mode=False) will update this table, and even then only h_mean, t_mean and loss will be updated 
        # run validation after initialization to get the state before training (at epoch 0)
        record = self.eval_and_validate(triples_set, valid_matrix, model)
        new_record = np.reshape(np.asarray([global_epoch]+record+[int(loss_sum)]), (1,6))
        results_table = np.append(results_table, new_record, axis=0)
        self.pickle_object(RESULTS_PATH, 'w', results_table)
        self.pickle_object(PLOT_RESULTS_PATH, 'w', results_table)    
        return results_table, new_record


    def get_trainer(self, loss):
 	# building training:
        # Stochastic Gradient Descent (SGD) with Adagrad for adaptive learning-rates 
        trainer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
        # alternatively:
	trainer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
	return trainer
