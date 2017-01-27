'''
Author of this implementation: Kavita Chopra (10.2016, version 1.0)

RUN THE CODE: 

TRAINING: 
- "python transe.py"
EVALUATION: 
- "python transe.py evaluate"


Input
- training data consisting of two sets: positive training data (true triples), and 
  negative training data, where for every positive triple either the head or tail is 
  replaced (corrupted) by a random entity 
- Hyperparameters: 
  embedding space dimension k, learning rate

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
- before training a new model:
    - meta-data-file with the customized configurations is created and saved in 'models/' directory
    - meta-data-file is updated each time training is resumed at a different time 
    - initial embedding is saved to disk for visualization purposes e.g. after dimensionality reduction through PCA
- at customizable intervals current model is saved to disk so that training may be continued in different sessions
- global- and hyper-parameters of the model can be configured in the params.py file

More details can be found in readme file in same directory.
'''


import numpy as np
import tensorflow as tf
import pickle 
import timeit
import os
import sys
import params
import input
from input import pickle_object
import eval

# set and get some model meta settings
model_name = 'transe'
dim = params.dim
dataset = params.dataset


PATHS, PLOT_PATHS = input.getPATHS(model_name, dim, dataset)
PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH = PATHS[0], PATHS[1], PATHS[2], PATHS[3], PATHS[4]
PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH = PLOT_PATHS[0], PLOT_PATHS[1]

# default setting for eval_mode, change if arg passed when running code
eval_mode = False
filtered = False

# sys.argv[0] is 'python',  sys.argv[1] is transe.py  now check for optional tags 
if len(sys.argv) >= 2:
	if sys.argv[1] == 'evaluate': 
	    eval_mode = True
	    if len(sys.argv)==3:
            	if sys.argv[2] == 'filtered':
	    		filtered = True 

# norm or distance measure (L1  or L2) for scoring function
def tensor_norm(tensor, l1_flag=True):
    if l1_flag:
        return tf.reduce_sum(tf.abs(tensor), reduction_indices=1)
    else: 
        return tf.sqrt(tf.reduce_sum(tf.square(tensor), reduction_indices=1))  # leave out sqrt for faster processing, since sqrt is monotonous function, not affecting the optimum in an optimization problem
	#return tf.reduce_sum(tf.square(tensor), reduction_indices=1)

def run_training(): 
    
    # set global- and hyper-parameters, for description of each param see file params.py

    shuffle_data = params.shuffle_data 
    check_collision = params.check_collision
    swap = params.swap
    device = params.device
    max_epoch = params.max_epoch
    margin = params.margin
    learning_rate = params.learning_rate              
    batch_size = params.batch_size
    l1_flag = params.l1_flag
    result_log_cycle = params.result_log_cycle
    test_size = params.test_size
    normalize_ent = params.normalize_ent
    
     
    # load set of all triples, train, valid and test data
    triples, train, valid, test = input.load_data(dataset, swap=swap)

    # load dicts (URI strings to int) 
    ent_URI_to_int, rel_URI_to_int = input.create_dicts(dataset, triples)    

    # load input-formats for script: triples_set (for faster existential checks) and int-matrices for triple stores (train-, test- and valid-set)
    triples_set, train_matrix, valid_matrix, test_matrix  = input.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    # entity and relation-lists:
    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations

    # load existing model (that is, model parameters) with given configurations or initialize new and save to disk
    if os.path.isfile(MODEL_PATH):
        print "\n\nExisting TransE model is being loaded...\n"
        model = pickle_object(MODEL_PATH, 'r')
        ent_array_map = model[0]
        rel_array_map = model[1]

        # if 'evaluate' tag was passed when running the script, only run evaluation on test-set, save top triples and terminate
	if eval_mode:	 
		top_triples = eval.run_evaluation(triples_set, test_matrix, ent_array_map, rel_array_map, score_func=input.score_func, l1_flag=l1_flag, test_size=test_size, eval_mode=True, filtered=filtered, verbose=True) 
                pickle_object(PATH + 'top_triples', 'w', top_triples)
		return
    else: 
        # case that no trained model with the given configurations exists, but eval_mode=True has been passed 
        if eval_mode:   
		print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(model_name)
		return

        # write model configurations to disk (meta-data on trained model)
        print "\n\nNew TransE model is being initialized and saved before training starts..."
        input.save_model_meta(model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, dim, learning_rate, normalize_ent, check_collision)
        ent_array_map, rel_array_map = input.init_params(n,m,dim)
        model = [ent_array_map,rel_array_map]
        pickle_object(INITIAL_MODEL, 'w', model)
        

    # open validation-results table to retrieve the last trained epoch
    # if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
        input.save_model_meta(model_name, MODEL_META_PATH, dim, learning_rate, normalize_ent, check_collision, global_epoch, resumed=True)
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits', 'total_loss']), (1,6))
	# note that only running validation (eval_mode=False) will update this table, and even then only h_mean, t_mean and loss will be updated 

        # run validation after initialization to get the state before training (at epoch 0)
        record = eval.run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, score_func=input.score_func, l1_flag=l1_flag, test_size=test_size)
        global_epoch = 0    
        new_record = np.reshape(np.asarray([global_epoch]+record+[0]), (1,6))
        results_table = np.append(results_table, new_record, axis=0)
        pickle_object(RESULTS_PATH, 'w', results_table)
        pickle_object(PLOT_RESULTS_PATH, 'w', results_table)
        
    # launch TF Session and build computation graph 
    # meta settings passed to the graph 

    g = tf.Graph()
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with g.as_default(), g.device('/'+device+':0'), tf.Session(config=config) as sess:
    '''
    with g.as_default(), g.device('/'+device+':0'), tf.Session() as sess: 

        # initialize model parameters (TF Variable) with numpy objects: entity matrix (n x dim) and relation matrix (m x dim)  
        E = tf.Variable(ent_array_map, name='E')
        R = tf.Variable(rel_array_map, name='R')

        # placeholders for current batch-input (int-arrays) in every gradient step
	# placeholders for true triples 
        h_ph = tf.placeholder(tf.int32, shape=(None))     #head (subject)
        t_ph = tf.placeholder(tf.int32, shape=(None))     #tail (object)

        # ph for common relation
        l_ph = tf.placeholder(tf.int32, shape=(None))     #label (relation)

        # ph for corrupted triple
        h_1_ph = tf.placeholder(tf.int32, shape=(None))      #head from corrupted counterpart triple 
        t_1_ph = tf.placeholder(tf.int32, shape=(None))      #tail from corrupted counterpart triple 
    
        # tf.gather for matrix-slicing based on E,R and respective placeholders
        h = tf.gather(E, h_ph) 
        l = tf.gather(R, l_ph) 
        t = tf.gather(E, t_ph) 
        h_1 = tf.gather(E, h_1_ph) 
        t_1 = tf.gather(E, t_1_ph) 
       
        # loss function: maximize margin between pos and neg triple
        loss = tf.reduce_sum(margin + tensor_norm((h + l) - t, l1_flag) - tensor_norm((h_1 + l) - t_1, l1_flag))
        #pos_score = tensor_norm((h + l) - t, l1_flag)
	#neg_score = tensor_norm((h_1 + l) - t_1, l1_flag)
        #loss = tf.reduce_sum(tf.maximum(tf.sub(pos_score, neg_score)+1, 0))
        # building training:
        # Stochastic Gradient Descent (SGD) with Adagrad for adaptive learning-rates 
        trainer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        # alternatively:
        # trainer=tf.train.AdamOptimizer().minimize(loss)
        
        # ops for normalizing E 
        norm = tf.sqrt(tf.reduce_sum(tf.square(E), 1, keep_dims=True))
        E_new = tf.div(E,norm)
	E_norm = tf.assign(E, E_new)

        
	print "\n****************Training of TransE starts!****************\n"
        print "Number of Triples in Training data: {}".format(len(train_matrix))
        print "Latent Embedding Dimension: {}".format(dim)
        train_batches = np.array_split(train_matrix, len(train_matrix)/batch_size)
        print "Batch size for Stochastic Gradient Descent: {}".format(len(train_batches[0]))
        if l1_flag: 
		print "Distance Measure: L1 norm\n"
	else: 
		print "Distance Measure: L2 norm\n"
        #vector X_id mirrors indices of train_matrix to allow inexpensive shuffling before each epoch
        X_id = np.arange(len(train_matrix))

        #op for Variable initialization 
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(max_epoch):
            print "\nepoch: {}".format(global_epoch)
            if shuffle_data: 
                np.random.shuffle(X_id)
            start = timeit.default_timer()
            # in each epoch accumulate loss from gradient steps 
            loss_sum = 0
            # split the training batch into subbatches; with array_split we will cover all triples even if resulting in uneven batch sizes 
            train_batches = np.array_split(train_matrix[X_id], len(train_matrix)/batch_size)
            for j in range(len(train_batches)):

                # get all input batches for current gradient step: 
                # extract h, l and t batches from positive (int) triple batch 
                pos_matrix = train_batches[j]
                h_batch, l_batch, t_batch = pos_matrix[:,0], pos_matrix[:,1], pos_matrix[:,2]
 
                # extract h_1, and t_1 batches from randomly created negative (int) triple batch 
                neg_matrix = input.corrupt_triple_matrix(triples_set, pos_matrix, n)
                h_1_batch, t_1_batch = neg_matrix[:,0], neg_matrix[:,2]

                # feed placeholders with current input batches 
                feed_dict={h_ph: h_batch, l_ph: l_batch, t_ph: t_batch, h_1_ph: h_1_batch, t_1_ph: t_1_batch} 
                _, loss_value = sess.run(([trainer, loss]), feed_dict=feed_dict)

                loss_sum += loss_value
            print "total loss of epoch: {}".format(loss_sum)
            # after an epoch decide to normalize entities 
            if normalize_ent: 
                sess.run(E_norm) 
                '''
                # check if normalization was successful: yes it was :)
                x = E.eval()
                print np.linalg.norm(x, axis=1)
                '''	     
            stop = timeit.default_timer()
            print "time taken for current epoch: {} sec".format((stop - start))
            global_epoch += 1
            if global_epoch > 752:
			test_size = None
			result_log_cycle = 25

            #validate model on valid_matrix and save current model after each result_log_cycle
            #if global_epoch == 1 or global_epoch == 10 or global_epoch%result_log_cycle == 0:
            if global_epoch%result_log_cycle == 0:
                # extract (numpy) parameters from updated TF variables 
		ent_array_map = E.eval()
                rel_array_map = R.eval()
                record = eval.run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, score_func=input.score_func, l1_flag=l1_flag, test_size=test_size)
                new_record = np.reshape(np.asarray([global_epoch]+record+[int(loss_sum)]), (1,6))
                # save model to disk only if both h_rank_mean and t_rank_mean improved 
                if min(results_table[1:len(results_table),1]) >= new_record[0,1] and min(results_table[1:len(results_table),2]) >= new_record[0,2]:
		        model = [ent_array_map, rel_array_map]
		        pickle_object(MODEL_PATH, 'w', model)
                # print validation results and save results to disk (to two directories where it is accessible for other application, e.g. plotting etc) 
                results_table = np.append(results_table, new_record, axis=0)
                pickle_object(RESULTS_PATH, 'w', results_table)
                pickle_object(PLOT_RESULTS_PATH, 'w', results_table)
                if global_epoch != max_epoch:
			print "\n\n******Continue Training******"

    
def main(arg=None):
    run_training()
    
if __name__=="__main__": 
    #tf.app.run()  # allows a TF-flag-passthrough 
    main()    
    
    
