import numpy as np
import tensorflow as tf
from KGC_Model import KGC_Model
from datetime import datetime
from math import sqrt
import eval



class TransE(KGC_Model):
    def __init__(self, dataset, swap, dim, margin, l1_flag, device, learning_rate, max_epoch, batch_size, test_size, result_log_cycle, shuffle_data=True, check_collision=True, normalize_ent=True):

        KGC_Model. __init__(self, dataset, swap, 'TransE', dim, shuffle_data=True, check_collision=True)
        self.margin = margin
        self.l1_flag = l1_flag

	self.device = device
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
	self.batch_size = batch_size
        self.test_size = test_size
        self.result_log_cycle = result_log_cycle
        self.normalize_ent = normalize_ent
     
    def normalize_embedding(self, x):	
        return np.array([x[i]/np.linalg.norm(x) for i in range(len(x))])   #consider axis=1 here

    #computes the distance score of a triple batch: x=(h+l-t)
    #used in validation part 
    #def score_func(h, l, t, self.l1_flag=True):
    def score_func(self, h, l, t):
        if self.l1_flag:
            return -1 * np.linalg.norm((x), ord=1, axis=1)
        else: 
            return -1 * np.linalg.norm(x, axis=1)

    def init_params(self, n,m):
        ent_array_map = np.random.uniform(-6/sqrt(self.dim),6/sqrt(self.dim), (n, self.dim))
        rel_array_map = np.random.uniform(-6/sqrt(self.dim),6/sqrt(self.dim), (m, self.dim))

        ent_array_map = self.normalize_embedding(ent_array_map)
        rel_array_map = self.normalize_embedding(rel_array_map)
        return ent_array_map, rel_array_map

    #method writes model configurations to disk 
    #def save_model_meta(self.model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, self.dim, self.learning_rate, self.normalize_ent, self.check_collision, global_epoch=None, resumed=False):
    def save_model_meta(self, MODEL_META_PATH, PLOT_MODEL_META_PATH, global_epoch=None, resumed=False):
        if resumed==False: 
            text_file = open(MODEL_META_PATH, "w")
            text_file.write("\nmodel: {}\n\n".format(self.model_name))

            text_file.write("created on: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
            text_file.write("embedding dimension:  {}\n".format(self.dim))
            text_file.write("learning rate:  {}\n".format(self.learning_rate))
            text_file.write("normalized entity vectors:  {}\n".format(self.normalize_ent))
            text_file.write("collision check:  {}\n".format(self.check_collision))
            text_file.close()

        if resumed==True: 
            new_lines = "\ntraining resumed on {}\nat epoch: {}\nwith learning rate: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), global_epoch, self.learning_rate)
            with open(MODEL_META_PATH, "a") as f:
                f.write(new_lines)

    def get_eval_tags(self, sys_tags): 
    # default setting for eval_mode, change if arg passed when running code
   	eval_mode = False
	filtered = False

	# sys.argv[0] is 'python',  sys.argv[1] is transe.py  now check for optional tags 
	if len(sys_tags) >= 2:
		if sys_tags[1] == 'evaluate': 
	    		eval_mode = True
	    		if len(sys_tags)==3:
            			if sys_tags[2] == 'filtered':
	    				filtered = True 
	return eval_mode, filtered 


    def load_model(self, MODEL_PATH):
        model = self.pickle_object(MODEL_PATH, 'r')
        ent_array_map = model[0]
        rel_array_map = model[1]
        return ent_array_map, rel_array_map

    def save_model(self, MODEL_PATH, ent_array_map, rel_array_map):
        model = [ent_array_map, rel_array_map]
        self.pickle_object(MODEL_PATH, 'w', model)


    def get_graph_variables(self, ent_array_map, rel_array_map): 
        # initialize model parameters (TF Variable) with numpy objects: entity matrix (n x dim) and relation matrix (m x dim)  
        E = tf.Variable(ent_array_map, name='E')
        R = tf.Variable(rel_array_map, name='R')
        return E, R

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

    def get_model_parameters(self, E, R, h_ph, l_ph, t_ph, h_1_ph, t_1_ph): 
        # tf.gather for matrix-slicing based on E,R and respective placeholders
        h = tf.gather(E, h_ph) 
        l = tf.gather(R, l_ph) 
        t = tf.gather(E, t_ph) 
        h_1 = tf.gather(E, h_1_ph) 
        t_1 = tf.gather(E, t_1_ph) 
 	return h, l, t, h_1, t_1

    def get_loss(self, h, l, t, h_1, t_1): 
	pos_score = self.tensor_norm((h + l) - t)
	neg_score = self.tensor_norm((h_1 + l) - t_1)
	return tf.reduce_sum(tf.maximum(tf.sub(pos_score, neg_score)+1, 0))
        #return tf.reduce_sum(self.margin + pos_score - neg_score)

    def normalize_entity_op(self, E):
        # ops for normalizing E 
        norm = tf.sqrt(tf.reduce_sum(tf.square(E), 1, keep_dims=True))
        E_new = tf.div(E,norm)
	E_norm = tf.assign(E, E_new)
        return E_norm


    def model_intro_print(self, train_matrix): 
	print "\n****************Training of TransE starts!****************\n"
        print "Number of Triples in Training data: {}".format(len(train_matrix))
        print "Latent Embedding Dimension: {}".format(self.dim)
        train_batches = np.array_split(train_matrix, len(train_matrix)/self.batch_size)
        print "Batch size for Stochastic Gradient Descent: {}".format(len(train_batches[0]))
        if self.l1_flag: 
		print "Distance Measure: L1 norm\n"
	else: 
		print "Distance Measure: L2 norm\n"

    # norm or distance measure (L1  or L2) for scoring function
    def tensor_norm(self, tensor):
        if self.l1_flag:
        	return tf.reduce_sum(tf.abs(tensor), reduction_indices=1)
        else: 
        	return tf.sqrt(tf.reduce_sum(tf.square(tensor), reduction_indices=1))  # leave out sqrt for faster processing, since sqrt is monotonous function, not affecting the optimum in an optimization problem
	#return tf.reduce_sum(tf.square(tensor), reduction_indices=1)

    def evaluate_model(self, PATH, triples_set, test_matrix, ent_array_map, rel_array_map, filtered):
	top_triples = eval.run_evaluation(triples_set, test_matrix, ent_array_map, rel_array_map, score_func=self.score_func, l1_flag=self.l1_flag, test_size=self.test_size, eval_mode=True, filtered=filtered, verbose=True) 
        self.pickle_object(PATH + 'top_triples', 'w', top_triples)

    def update_results_table(self, RESULTS_PATH, PLOT_RESULTS_PATH, triples_set, valid_matrix, ent_array_map, rel_array_map, global_epoch, loss_sum, results_table=None, init=False): 
	if init:
        	results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits', 'total_loss']), (1,6))
		# note that only running validation (eval_mode=False) will update this table, and even then only h_mean, t_mean and loss will be updated 
        # run validation after initialization to get the state before training (at epoch 0)
        record = eval.run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, score_func=self.score_func, l1_flag=self.l1_flag, test_size=self.test_size)
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

