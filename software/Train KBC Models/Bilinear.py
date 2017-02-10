import numpy as np
import tensorflow as tf
from datetime import datetime
import timeit
from KBC_Class import KBC_Class
import os
import sys
sys.path.insert(0,'eval_files/bilinear_eval_files')
import np_eval
import tf_eval
import np_eval1


# Class Bilinear inherits from KBC_Class
# In this class all model-specific data and methods are added
# one model-specific state added in the constructor of this class is the tag 'diagonal' which denotes if the relation matrix Mr is a diagonal matrix

class Bilinear(KBC_Class):

    def __init__(self, dataset, swap, dim, margin, device, memory, learning_rate, max_epoch, batch_size, test_size, result_log_cycle, diagonal,  eval_with_np=True, shuffle_data=True, check_collision=True, normalize_ent=True):

	self.diagonal = diagonal

	if self.diagonal: 
        	model_name = 'Diagonal'
	else: 
		model_name = 'Bilinear'
        KBC_Class. __init__(self, dataset, swap, model_name, dim, margin, device, memory, learning_rate, max_epoch, batch_size, test_size, result_log_cycle, eval_with_np=True, shuffle_data=True, check_collision=True, normalize_ent=True)


  
    # numpy and tensorflow methods for computing the score of the model 
    def np_score2(self, h, l, t):
    	score = np.dot(np.dot(h,l), (np.transpose(t)))
    	return score 
 

    # score-function for triple batches with same label l: 
    def np_score(self, h, l, t):
    	scores = []
    	for i in range(len(h)): 
        	scores.append( np.dot(np.dot(h[i],l), (np.transpose(t[i]))) )
    	return np.array(scores)


    def tf_score(self, h,l,t):
	if self.diagonal:
        	return tf.batch_matmul(tf.batch_matmul(h,tf.matrix_diag(l, name=None)), tf.transpose(t, perm=[0, 2, 1]))
        else: 
        	return tf.batch_matmul(tf.batch_matmul(h,l), tf.transpose(t, perm=[0, 2, 1]))


    #initialize model parameters W and Mr
    def init_params(self, n, m): 
        if self.diagonal:
        	Mr = np.random.rand(m, self.dim)
        else: 
        	Mr = np.random.rand(m, self.dim, self.dim)
    	W = np.random.rand(n,1,self.dim)
    	return [W, Mr]


    #method writes model configurations to disk - contains general and model-specific settings
    def save_model_meta(self, MODEL_META_PATH, PLOT_MODEL_META_PATH, global_epoch=None, resumed=False):
        if resumed==False: 
            text_file = open(MODEL_META_PATH, "w")
            text_file.write("\n******** model: {} ********\n\n\n".format(self.model_name))
	    text_file.write("trained on: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
	    text_file.write("dataset:  {}\n".format(self.dataset))

	    text_file.write("\n*** general settings ***\n\n")
            text_file.write("embedding dimension:  {}\n".format(self.dim))
            text_file.write("learning rate:  {}\n".format(self.learning_rate))
            text_file.write("normalize entity vectors:  {}\n".format(self.normalize_ent))
            text_file.write("collision check:  {}\n".format(self.check_collision))
		
	    text_file.write("\n*** model-specific settings ***\n\n")
	    # add here model-specific settings

            text_file.close()
        if resumed==True: 
            new_lines = "\ntraining resumed on {}\nat epoch: {}\nwith learning rate: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), global_epoch, self.learning_rate)
            with open(MODEL_META_PATH, "a") as f:
                f.write(new_lines)



    def get_graph_variables(self, model): 
        # initialize model parameters (TF Variable) with numpy objects: entity matrix (n x dim) and relation matrix (m x dim)  
        E = tf.Variable(model[0], name='E')
        R = tf.Variable(model[1], name='R')
        return E, R


    def normalize_entity_op(self, E):
        norm = tf.sqrt(tf.reduce_sum(tf.square(E), 2, keep_dims=True))
        E_new = tf.div(E,norm)
	E_norm = tf.assign(E, E_new)
        return E_norm

    # based on placeholders and learnable TF variables, apply matrix slicing on variables using gather: 
    # major advantage in this implementation under the score-based framework: plug in int vectors, no matter what the variable dimensions are
    def get_model_parameters(self, E, R, h_ph, l_ph, t_ph, h_1_ph, t_1_ph): 
        h = tf.gather(E, h_ph) 
        l = tf.gather(R, l_ph) 
        t = tf.gather(E, t_ph) 
        h_1 = tf.gather(E, h_1_ph) 
        t_1 = tf.gather(E, t_1_ph) 
 	return h, l, t, h_1, t_1


    def get_scores(self, h, l, t, h_1, t_1):
	pos_score = self.tf_score(h, l, t)
	neg_score = self.tf_score(h_1, l, t_1)
	return pos_score, neg_score


    def adapt_params_for_eval(self, model):
	W_param = model[0]
	Mr_param = model[1]
     	m = len(Mr_param)
    	W_eval_param =  np.reshape(W_param, (W_param.shape[0], W_param.shape[2]))
    	#Mr_param = np.array([np.dot(A_param[i], np.transpose(B_param[i])) for i in range(m)])
    	if self.diagonal: 
        	Mr_eval_param = {}
        	for i in range(m): 
            		Mr_eval_param[i] = np.diag(Mr_param[i])
        	return [W_eval_param, Mr_eval_param]
    	return [W_eval_param, Mr_param]


    def eval_and_validate2(self, triples_set, test_matrix, model, filtered = False, eval_mode = False):
	model = self.adapt_params_for_eval(model)
	if eval_mode: 
		top_triples = eval.run_evaluation(triples_set, test_matrix, model, score_func=self.np_score, test_size=self.test_size, eval_mode=True, filtered=filtered, verbose=True)
		return top_triples
	record = eval.run_evaluation(triples_set, test_matrix, model, score_func=self.np_score, test_size=self.test_size, filtered=filtered)
	return record


    def eval_and_validate(self, triples_set, test_matrix, model, filtered = False, eval_mode = False):
	
	eval_with_np = True
	if eval_with_np: 
		score_func = self.np_score
		eval = np_eval
		model = self.adapt_params_for_eval(model)
	else: 
		score_func = self.tf_score
		eval = tf_eval
	if eval_mode: 
		top_triples = eval.run_evaluation(triples_set, test_matrix, model, score_func=score_func, eval_mode=True, filtered=filtered, verbose=True)
		return top_triples
	record = eval.run_evaluation(triples_set, test_matrix, model, score_func=score_func, test_size=self.test_size, filtered=filtered)
	return record
 

    def run_training(self, PATHS, PLOT_PATHS, n, m, eval_mode, filtered, triples_set, train_matrix, valid_matrix, test_matrix): 

	    # get required paths (all based on ALL_DATA_PATH defined in KBC_Data
	    PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH = PATHS[0], PATHS[1], PATHS[2], PATHS[3], PATHS[4]
	    PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH = PLOT_PATHS[0], PLOT_PATHS[1]

	    # load existing model (that is, model parameters) with given configurations or initialize new and save to disk
	    if os.path.isfile(MODEL_PATH):
		print "\n\nExisting {} model is being loaded...\n".format(self.model_name)
		model = self.load_model(MODEL_PATH)

		# if 'evaluate' tag was passed when running the script, only run evaluation on test-set, save top triples and terminate
		if eval_mode:
			top_triples = self.eval_and_validate(triples_set, test_matrix, model, filtered=filtered, eval_mode=eval_mode)
			if filtered: 
				self.pickle_object(PATH + 'top_triples', 'w', top_triples)
			return
	    else: 
		# case that no trained model with the given configurations exists, but eval_mode=True has been passed 
		if eval_mode:   
			print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(self.model_name)
			return

		# write model configurations and initial model to disk (meta-data on trained model)
		print "\n\nNew {} model is being initialized and saved before training starts...".format(self.model_name)
		self.save_model_meta(MODEL_META_PATH, PLOT_MODEL_META_PATH)
		model = self.init_params(n,m)
		self.save_model(INITIAL_MODEL, model)
		
	    # open validation-results table to retrieve the last trained epoch
	    # if it does not exist, create a new result_table 
	    if os.path.isfile(RESULTS_PATH):
		results_table = self.pickle_object(RESULTS_PATH, 'r')
		global_epoch = int(results_table[-1][0]) #update epoch_num
		self.save_model_meta(MODEL_META_PATH, global_epoch, resumed=True)
	    else:
		global_epoch = 0
		results_table, new_record = self.update_results_table(RESULTS_PATH, PLOT_RESULTS_PATH, triples_set, valid_matrix, model, global_epoch, 0, init=True)


	    # launch TF Session and build computation graph 
	    # meta settings passed to the graph 
	    g = tf.Graph()
	    config = tf.ConfigProto()
	    config.gpu_options.per_process_gpu_memory_fraction = self.memory
	    with g.as_default(), g.device('/'+self.device+':0'), tf.Session(config=config) as sess:
	    #with g.as_default(), g.device('/'+self.device+':0'), tf.Session() as sess: 

		E, R = self.get_graph_variables(model)

		h_ph, l_ph, t_ph, h_1_ph, t_1_ph = self.get_graph_placeholders()
		h, l, t, h_1, t_1 = self.get_model_parameters(E, R, h_ph, l_ph, t_ph, h_1_ph, t_1_ph)

		pos_score, neg_score = self.get_scores(h, l, t, h_1, t_1)
	
		loss = self.get_loss(pos_score, neg_score)
	  
		trainer = self.get_trainer(loss)
	 
		self.model_intro_print(train_matrix)
	
		#op for Variable initialization 
		#init_op = tf.global_variables_initializer()
	 	init_op = tf.initialize_all_variables()
		sess.run(init_op)
	  
		#vector X_id mirrors indices of train_matrix to allow inexpensive shuffling before each epoch
		X_id = np.arange(len(train_matrix))

		for i in range(self.max_epoch):
		    print "\nepoch: {}".format(global_epoch)
		    if self.shuffle_data: 
		        np.random.shuffle(X_id)
		    start = timeit.default_timer()
		    
		    loss_sum = 0
		    # split the training batch into subbatches; with array_split we will cover all triples even if resulting in uneven batch sizes 
		    train_batches = np.array_split(train_matrix[X_id], len(train_matrix)/self.batch_size)
		    for j in range(len(train_batches)):

		        # get all input batches for current gradient step: 

		        # extract h, l and t batches from positive (int) triple batch 
		        pos_matrix = train_batches[j]
		        h_batch, l_batch, t_batch = pos_matrix[:,0], pos_matrix[:,1], pos_matrix[:,2]
	 
		        # extract h_1, and t_1 batches from randomly created negative (int) triple batch 
		        neg_matrix = self.corrupt_triple_matrix(triples_set, pos_matrix, n)
		        h_1_batch, t_1_batch = neg_matrix[:,0], neg_matrix[:,2]

		        # feed placeholders with current input batches 
		        feed_dict={h_ph: h_batch, l_ph: l_batch, t_ph: t_batch, h_1_ph: h_1_batch, t_1_ph: t_1_batch} 
		        _, loss_value = sess.run(([trainer, loss]), feed_dict=feed_dict)

		        loss_sum += loss_value
		    print "average loss/error per triple: {}".format(float(loss_sum)/len(train_matrix))
		    # after an epoch decide to normalize entities 
		    if self.normalize_ent: 
		        sess.run(self.normalize_entity_op(E)) 
		        ''' 
               		x = E.eval()
                	print x.shape
                	x = np.reshape(x, (n, 20))
                	print np.linalg.norm(x, axis=1)
                	'''     
		    stop = timeit.default_timer()
		    print "time taken for current epoch: {} sec".format((stop - start))
		    global_epoch += 1
		    if global_epoch > 500:
				self.test_size = None
				self.result_log_cycle = 25

		    #validate model on valid_matrix and save current model after each result_log_cycle
		    #if global_epoch == 1 or global_epoch == 10 or global_epoch%result_log_cycle == 0:
		    if global_epoch % self.result_log_cycle == 0:
		        # extract (numpy) parameters from updated TF variables 
			model = [E.eval(), R.eval()]
		        results_table, new_record = self.update_results_table(RESULTS_PATH, PLOT_RESULTS_PATH, triples_set, valid_matrix, model, global_epoch, loss_sum, results_table)
		        # save model to disk only if both h_rank_mean and t_rank_mean improved 
			if len(results_table) > 3:
		        	if min(np.array(results_table[1:len(results_table)-1,1], dtype=np.int32)) and min(np.array(results_table[1:len(results_table)-2,2], dtype=np.int32)):
					self.save_model(MODEL_PATH, model)
		        # print validation results and save results to disk (to two directories where it is accessible for other application, e.g. plotting etc)
		        if global_epoch != self.max_epoch:
				print "\n\n******Continue Training******"
