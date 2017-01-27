'''
Author of this implementation: Kavita Chopra (2016, version 1.0)

RUN THE CODE: 

TRAINING: 
- "python bilinear.py" for training with non-diagonal Mr 
- "python bilinear.py diagonal" to train the bilinear diagonal model
EVALUATION: 
- "python bilinear.py evaluate" for evaluation with non-diagonal Mr 
- "python bilinear.py diagonal evaluate" to evaluate the bilinear diagonal model

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
import timeit 
import pickle 
import os
import params
import input
from input import pickle_object
import eval
import sys


#default settings for model_meta, altered through flags passed when running the code 
diagonal = False
model_name ='bilinear'
eval_mode = False
filtered = False

# sys.argv[0] is 'python',  sys.argv[1] is transe.py  now check for optional tags 
if len(sys.argv)==2:
	if sys.argv[1] == 'evaluate': 
	    eval_mode = True
	if sys.argv[1] == 'diagonal': 
	    diagonal = True
	    model_name = 'diagonal'
if len(sys.argv)==3:
	if sys.argv[1] == 'diagonal' and sys.argv[2] == 'evaluate': 
	    diagonal = True
	    model_name = 'diagonal'
            eval_mode = True

if len(sys.argv) > 2:
	if sys.argv[1] == 'evaluate': 
	    eval_mode = True
	    if len(sys.argv)==3:
            	if sys.argv[2] == 'filtered':
	    		filtered = True   

dim = params.n_red
dataset = params.dataset
normalize_ent = params.normalize_ent

PATHS, PLOT_PATHS = input.getPATHS(model_name, dim, dataset)
PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH = PATHS[0], PATHS[1], PATHS[2], PATHS[3], PATHS[4]
PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH = PLOT_PATHS[0], PLOT_PATHS[1]


#methods for building the tensor network 

#score: x0 * Mr * x1
def bilinear(h, t, h_1, t_1, Mr): 
    #tf.batch_matmul(tf.batch_matmul(h,M), tf.transpose(t, perm=[0, 2, 1]))
    if diagonal:
        pos_score = tf.batch_matmul(tf.batch_matmul(h,tf.matrix_diag(Mr, name=None)), tf.transpose(t, perm=[0, 2, 1]))  #[none, none] = [none, n_red] * [n_red, n_red] * [n_red, none]
        neg_score = tf.batch_matmul(tf.batch_matmul(h_1,tf.matrix_diag(Mr, name=None)), tf.transpose(t_1, perm=[0, 2, 1]))
    else: 
        pos_score = tf.batch_matmul(tf.batch_matmul(h,Mr), tf.transpose(t, perm=[0, 2, 1]))  #[none, none] = [none, n_red] * [n_red, n_red] * [n_red, none]
        neg_score = tf.batch_matmul(tf.batch_matmul(h_1,Mr), tf.transpose(t_1, perm=[0, 2, 1]))
        
    return pos_score, neg_score



def run_training(model_name): 
    
    #set global- and hyper-parameters, for description of each param see file params.py

    n_red = params.n_red
    shuffle_data = params.shuffle_data 
    check_collision = params.check_collision
    swap = params.swap
    device = params.device 
    max_epoch = params.max_epoch
    learning_rate = params.learning_rate              
    batch_size = params.batch_size
    result_log_cycle = params.result_log_cycle 
    corrupt_two = params.corrupt_two 
    test_size = params.test_size
    valid_verbose = params.valid_verbose
    train_verbose = params.train_verbose
    normalize_ent = params.normalize_ent
 

    if eval_mode: 
	mode = 'evaluated'
    else: 
        mode = 'trained'

    if model_name == 'diagonal':
	print "\n\nBilinear model is {} with diagonal relation matrices.\n".format(mode)  
    else:
    	print "\n\nBilinear model is {} with non-diagonal relation matrices.\n".format(mode)


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
        print "\n\nExisting model is being loaded...\n"
        model = pickle_object(MODEL_PATH, 'r')
        ent_array_map = model[0]
        rel_array_map = model[1]

        # if 'evaluate' tag was passed when running the script, only run evaluation on test-set, save top triples and terminate
        if eval_mode:    
		W_eval_param, Mr_eval_param = input.adapt_params_for_eval(diagonal, ent_array_map, rel_array_map)
		top_triples = eval.run_evaluation(triples_set, test_matrix, W_eval_param, Mr_eval_param, score_func=input.score_func, eval_mode=True, filtered=filtered, verbose=True, test_size=test_size) 
		pickle_object(PATH + 'top_triples', 'w', top_triples)
		return
    else: 
        # case that no trained model with the given configurations exists, but eval_mode=True has been passed 
        if eval_mode:   
        	print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(model_name)
                return

        # write model configurations to disk (meta-data on trained model)
        print "\n\nNew Bilinear model is being initialized and saved before training starts...\n"
        input.save_model_meta(model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, n_red, learning_rate, corrupt_two, normalize_ent, check_collision)
        ent_array_map, rel_array_map = input.init_params(diagonal, n, m, n_red)
        eval_rel_array_map = rel_array_map
        model = [ent_array_map,rel_array_map]
        pickle_object(INITIAL_MODEL, 'w', model)
        

    # open validation-results table to retrieve the last trained epoch
    # if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
        input.save_model_meta(model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, n_red, learning_rate, corrupt_two, normalize_ent, check_collision, global_epoch, resumed=True)
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits', 'total_loss']), (1,6))
    # note that only running validation (eval_mode=False) will update this table, and even then only h_mean, t_mean and loss will be updated 

        # run validation after initialization to get the state before training (at epoch 0)
        W_eval_param, Mr_eval_param = input.adapt_params_for_eval(diagonal, ent_array_map, rel_array_map)
        record = eval.run_evaluation(triples_set, valid_matrix, W_eval_param, Mr_eval_param, score_func=input.score_func, test_size=test_size)
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


        # initialize model parameters (TF Variable) with numpy objects 
        W = tf.Variable(ent_array_map, name='W')
        M = tf.Variable(rel_array_map, name='Mr')
  
        # placeholders for current batch-input (int-arrays) in every gradient step
        # placeholders for true triples 
        h_ph = tf.placeholder(tf.int32, shape=(None))     #head (subject)
        t_ph = tf.placeholder(tf.int32, shape=(None))     #tail (object)

        # ph for common relation
        l_ph = tf.placeholder(tf.int32, shape=(None))     #label (relation)

        # ph for corrupted triple
        h_1_ph = tf.placeholder(tf.int32, shape=(None))      #head from corrupted counterpart triple 
        t_1_ph = tf.placeholder(tf.int32, shape=(None))      #tail from corrupted counterpart triple 

        # tf.gather-ops for matrix and tensor-slicing on W and M based on current placeholder values  
	h = tf.gather(W, h_ph)
       	Mr = tf.gather(M, l_ph)
        t = tf.gather(W, t_ph)
        h_1 = tf.gather(W, h_1_ph)
        t_1 = tf.gather(W, t_1_ph)

    	pos_score, neg_score = bilinear(h, t, h_1, t_1, Mr)
    	loss = tf.reduce_sum(tf.maximum(tf.sub(neg_score, pos_score)+1, 0))
        margin = 2
        # loss = tf.reduce_sum(margin + neg_score - pos_score)
        # building training: 
        # SGD with Adagrad opimizer using adaptive learning rates 
    	trainer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    	# alternatively
   	    # trainer=tf.train.AdamOptimizer().minimize(loss)              
 
        # ops for normalizing W
    	norm = tf.sqrt(tf.reduce_sum(tf.square(W), 2, keep_dims=True))
    	W_new = tf.div(W,norm)
    	W_norm = tf.assign(W, W_new)            
    
        print "\n****************Training of Bilinear Model starts!****************\n"
        print "Number of Triples in Training data: {}".format(len(train_matrix))
        print "Latent Embedding Dimension (Factorization Rank): {}".format(n_red)
        train_batches = np.array_split(train_matrix, len(train_matrix)/batch_size)
        print "Batch size for Stochastic Gradient Descent: {}".format(len(train_batches[0]))
        
        #vector X_id mirrors indices of train_matrix to allow inexpensive shuffling before each epoch
        X_id = np.arange(len(train_matrix))

        # determine the size of the training data and #iterations for training loop based batch_size and corrupt_two-tag 
	x = 1 
	if corrupt_two: 
		x = 2
	batch_num = (x * len(train_matrix))/batch_size
	batch_size = int(batch_size/float(x))

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
                h_batch, l_batch, t_batch  = pos_matrix[:,0], pos_matrix[:,1], pos_matrix[:,2]  
                if corrupt_two:
                    h_batch = np.append(h_batch, h_batch, axis=0)
                    t_batch = np.append(t_batch, t_batch, axis=0)  
                    l_batch = np.append(l_batch, l_batch, axis=0)           
                neg_matrix = input.create_corrupt_matrix(triples_set, corrupt_two, pos_matrix, n, check_collision)
                h_1_batch, t_1_batch = neg_matrix[:,0], neg_matrix[:,2]

                # feed current int-input-batches to placeholders
                feed_dict={h_ph: h_batch, l_ph: l_batch, t_ph: t_batch, h_1_ph: h_1_batch, t_1_ph: t_1_batch} 
                _, loss_value = sess.run(([trainer, loss]), feed_dict=feed_dict)

                if train_verbose:
                    print loss_value
                loss_sum += loss_value
            print "total loss of epoch: {}".format(loss_sum)
            # after an epoch decide to normalize entities 
            if normalize_ent: 
                sess.run(W_norm) 
                ''' 
                x = W.eval()
                print x.shape
                x = np.reshape(x, (n, 20))
                print np.linalg.norm(x, axis=1)
                '''    
            stop = timeit.default_timer()
            print "time taken for current epoch: {} sec".format((stop - start))
            global_epoch += 1
            
            if global_epoch > 751:
                test_size = None
                result_log_cycle = 25

            #validate model on valid_matrix and save current model after each result_log_cycle
            #if global_epoch == 1 or global_epoch == 10 or global_epoch%result_log_cycle == 0:
            if global_epoch%result_log_cycle == 0:
                # extract (numpy) parameters from updated TF variables 
                ent_array_map = W.eval()
                rel_array_map = M.eval()
                W_eval_param, Mr_eval_param = input.adapt_params_for_eval(diagonal, ent_array_map, rel_array_map)
                record = eval.run_evaluation(triples_set, valid_matrix, W_eval_param, Mr_eval_param, score_func = input.score_func, test_size=test_size, verbose=valid_verbose) 
                new_record = np.reshape(np.asarray([global_epoch]+record+[int(loss_sum)]), (1,6))
                # save model to disk only if both h_rank_mean and t_rank_mean improved 
                if test_size == None and min(results_table[1:len(results_table),1]) >= new_record[0,1] and min(results_table[1:len(results_table),2]) >= new_record[0,2]:
                	model = [ent_array_map, rel_array_map]
                	pickle_object(MODEL_PATH, 'w', model)
                	# print validation results and save results to disk (to two directories where it is accessible for other application, e.g. plotting etc) 
                results_table = np.append(results_table, new_record, axis=0)
                pickle_object(RESULTS_PATH, 'w', results_table)
               	pickle_object(PLOT_RESULTS_PATH, 'w', results_table)
                if global_epoch != max_epoch:
            		print "\n\n******Continue Training******"

    
def main(arg=None):
    run_training(model_name)
    
if __name__=="__main__": 
    #tf.app.run()  # allows a TF-flag-passthrough 
    main()
