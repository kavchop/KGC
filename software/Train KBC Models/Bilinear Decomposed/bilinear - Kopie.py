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
from datetime import datetime
import timeit 
import pickle 
import os
import zipfile 
import params
import input
import eval
import sys



os.chdir(os.getcwd())


diagonal = False
model_name ='bilinear_decomp'

eval_mode = False

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
  

dim = params.n_red
dataset = params.dataset

os.chdir(os.getcwd())

PATH = '../../../data/Trained Models/'+model_name+'/' + dataset + '/dim = '+str(dim) +'/'
if not os.path.exists(PATH):
    os.makedirs(PATH)

PLOT_PATH = '../../../data/Model Validation Results for Plotting/' + dataset + '/dim = '+str(dim) +'/'
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)


normalize_ent = params.normalize_ent
if not normalize_ent: 
    model_name = model_name + ' not normalized'

MODEL_PATH = PATH + model_name
INITIAL_MODEL = PATH + model_name + '_initial_model'
RESULTS_PATH = PATH + model_name + '_results'
MODEL_META_PATH = PATH + model_name + '_model_meta.txt'

PLOT_RESULTS_PATH = PLOT_PATH + model_name + '_results'
PLOT_MODEL_META_PATH = PLOT_PATH + model_name + '_model_meta.txt'


#methods for building the tensor network 

#score: x0 * Mr * x1
def bilinear(h, t, h_1, t_1, A, B): 

    pos_score = tf.batch_matmul(tf.batch_matmul(h,tf.batch_matmul(A, tf.transpose(B, perm=[0, 2, 1]))), tf.transpose(t, perm=[0, 2, 1]))
    neg_score = tf.batch_matmul(tf.batch_matmul(h_1,tf.batch_matmul(A, tf.transpose(B, perm=[0, 2, 1]))), tf.transpose(t_1, perm=[0, 2, 1]))
        
    return pos_score, neg_score



def run_training(model_name): 
    
    #set global- and hyper-parameters, for description of each param see file params.py

    n_red = params.n_red
    a = params.a
    shuffle_data = params.shuffle_data 
    check_collision = params.check_collision
    swap = params.swap
    max_epoch = params.max_epoch
    global_epoch = params.global_epoch
    learning_rate = params.learning_rate              
    batch_size = params.batch_size
    result_log_cycle = params.result_log_cycle
    embedding_log_cycle = params.embedding_log_cycle     
    corrupt_two = params.corrupt_two 
    valid_size = params.valid_size
    valid_verbose = params.valid_verbose
    train_verbose = params.train_verbose
    #normalize_ent = params.normalize_ent
 

    if eval_mode: 
	mode = 'evaluated'
    else: 
        mode = 'trained'

    if model_name == 'diagonal':
	print "\nBilinear model is {} with diagonal relation matrices.\n".format(mode)  
    else:
    	print "\nBilinear model is {} with non-diagonal relation matrices.\n".format(mode)


    #load set of all triples, train, valid and test data
    triples, train, valid, test = input.load_data(swap)
    ent_URI_to_int, rel_URI_to_int = input.create_dicts(triples)   #load dicts 
    
    #load input-formats: int-matrices and triples-set for faster search
    triples_set, train_matrix, valid_matrix, test_matrix  = input.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations

    #load existing model or initialize new and save to disk
    print MODEL_PATH
    if os.path.isfile(MODEL_PATH):
        print "\nExisting model is being loaded...\n"
        bilinear_model = input.pickle_object(MODEL_PATH, 'r')
        W_param = bilinear_model[0]
        print W_param.shape
        W_param = np.reshape(W_param, (W_param.shape[0],1, W_param.shape[1]))
        A_param = np.array(bilinear_model[1], dtype=np.float32)
        B_param = np.array(bilinear_model[2], dtype=np.float32)
        print A_param.shape, B_param.shape
	entity_embed = input.learned_ent_embed(W_param) 
	# if 'evaluate' tag passed when running script, only run evaluation on test_matrix and terminate 
	if eval_mode:	 
                Mr_param = np.array([np.dot(A_param[i], np.transpose(B_param[i])) for i in range(m)])	 
		eval.run_evaluation(diagonal, triples_set, test_matrix, entity_embed, Mr_param, eval_mode=True, verbose=True, test_size=valid_size) 
		return
    else: 
        if eval_mode: 
		print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(model_name)
		return	
        #write model configurations to disk
        print "\nNew model is being initialized and saved before training starts...\n"
        input.save_model_meta(model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, n_red, a, learning_rate, corrupt_two, normalize_ent, check_collision)
        W_param, A_param, B_param = input.init_params(m, n, n_red, a)
        bilinear_model = [W_param, A_param, B_param]
        input.pickle_object(INITIAL_MODEL, 'w', bilinear_model)

    entity_embed = input.learned_ent_embed(W_param) 
    #open eval-results table to retrieve the last trained global epoch
    #if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = input.pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
	input.save_model_meta(model_name, MODEL_META_PATH, PLOT_MODEL_META_PATH, n_red, a, learning_rate, corrupt_two, normalize_ent, check_collision, global_epoch, resumed=True)
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits', 'total_loss'], dtype=object), (1,6))
        #run evaluation after initialization to get the state before training (at global_epoch 0)
        Mr_param = np.array([np.dot(A_param[i], np.transpose(B_param[i])) for i in range(m)])	 
        record = eval.run_evaluation(diagonal, triples_set, valid_matrix, entity_embed, Mr_param, test_size=valid_size)  
        new_record = np.reshape(np.asarray([global_epoch]+record+[0]), (1,6))
        print "validation result of current embedding: {}\n".format(new_record)
        results_table = np.append(results_table, new_record, axis=0)
        input.pickle_object(RESULTS_PATH, 'w', results_table)
        input.pickle_object(PLOT_RESULTS_PATH, 'w', results_table)

    
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    #with tf.Session(config=config) as sess:
    # launch TF Session and build computation graph 
    # start a TF session and build the computation Graph
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:  
    #with tf.Session() as sess:

        # initialize model parameters (TF Variable) with numpy objects 
        A = tf.Variable(A_param, name='A')
        B = tf.Variable(B_param, name='B')
        W = tf.Variable(W_param, name='W')
  
        # placeholders for current input batch in a gradient step  
        h_ph = tf.placeholder(tf.int32, shape=(None))     #head (subject)
        l_ph = tf.placeholder(tf.int32, shape=(None))     #label (relation/predicate) 
        t_ph = tf.placeholder(tf.int32, shape=(None))     #tail (object)
         
        h_1_ph = tf.placeholder(tf.int32, shape=(None))      #head from corrupted counterpart triple 
        t_1_ph = tf.placeholder(tf.int32, shape=(None))      #tail from corrupted counterpart triple 
  

        # tf.gather-ops for matrix and tensor-slicing on W and M based on current placeholder values  
	h = tf.gather(W, h_ph)
       	A_ = tf.gather(A, l_ph)
        B_ = tf.gather(B, l_ph)
        t = tf.gather(W, t_ph)
        h_1 = tf.gather(W, h_1_ph)
        t_1 = tf.gather(W, t_1_ph)

    	pos_score, neg_score = bilinear(h, t, h_1, t_1, A_, B_)
    	loss = tf.reduce_sum(tf.maximum(tf.sub(neg_score, pos_score)+1, 0))

        # building training: 
        # SGD with Adagrad opimizer using adaptive learning rates 
    	trainer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    	# alternatively
   	# trainer=tf.train.AdamOptimizer().minimize(loss)              
 
        #ops for normalizing W
    	norm = tf.sqrt(tf.reduce_sum(tf.square(W), 2, keep_dims=True))
    	W_new = tf.div(W,norm)
    	W_norm = tf.assign(W, W_new)            
    
    	# op for Variable initialization 
   	init_op = tf.global_variables_initializer()

    	# vector X_id mirrors indices of train_matrix to allow inexpensive shuffling before each epoch
    	X_id = np.arange(len(train_matrix))               
    
        # determine the size of the training data and #iterations for training loop based batch_size and corrupt_two-tag 
	x = 1 
	if corrupt_two: 
		x = 2
	batch_num = (x * len(train_matrix))/batch_size
	batch_size = int(batch_size/float(x))

    	print "\nNumber of Triples in Training data: {}".format(len(train_matrix))
    	print "Iteration over training batches by relations (# {}) and maximal batch-size of {}".format(m, batch_size)

	# run initializer-op for variables 
	sess.run(init_op)
 
        print "\nTraining of Bilinear Model starts!"
        for i in range(max_epoch):
            print "\nepoch: {}".format(i)
            if shuffle_data: 
                np.random.shuffle(X_id)
            start = timeit.default_timer()
            loss_sum = 0
            train_batches = np.array_split(train_matrix[X_id], len(train_matrix)/batch_size)
            for j in range(len(train_batches)):
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
            #after completing one epoch print current loss:
            print "total loss of epoch: {}".format(loss_sum)
            
            if normalize_ent: 
                sess.run(W_norm) 
                ''' 
                x = W.eval()
                print x.shape
                x = np.reshape(x, (n, 20))
                print np.linalg.norm(x, axis=1)
                '''
 
            stop = timeit.default_timer()
            print "time taken for current epoch: {} min".format((stop - start)/60)
            global_epoch += 1
            '''
            #save model after each embedding_log_cycle
            if global_epoch%embedding_log_cycle == 0:
                W_param = W.eval()
                Mr_param = M.eval()
                bilinear_model = [W_param, A_param, B_param]
                input.pickle_object(MODEL_PATH, 'w', bilinear_model)
            '''
            if global_epoch > 500:
			valid_size = None
			embedding_log_cycle, result_log_cycle = 25, 25
            print A.eval()[1,:]
            #run validation on current embedding applied on validation set 
            if global_epoch == 1 or global_epoch == 10 or global_epoch%result_log_cycle == 0:   
                entity_embed = input.learned_ent_embed(W_param) 
                Mr_param = np.array([np.dot(A_param[i], np.transpose(B_param[i])) for i in range(m)])	 
                record = eval.run_evaluation(diagonal, triples_set, valid_matrix, entity_embed, Mr_param, test_size=valid_size, verbose=valid_verbose) 
                new_record = np.reshape(np.asarray([global_epoch]+record+[int(loss_sum)]), (1,6))
                if valid_size == None and min(results_table[1:len(results_table),1]) >= new_record[0,1] and min(results_table[1:len(results_table),2]) >= new_record[0,2]:
			#input.pickle_object('models/'+model_name+'_best_model', 'w', bilinear_model)
                    W_param = W.eval()
                	Mr_param = M.eval()
                	bilinear_model = [W_param, A_param, B_param]
                	input.pickle_object(MODEL_PATH, 'w', bilinear_model)
                #TODO: LINES FOR EARLY STOPPING  (a[1:-1,1]).tolist()
                print "validation result of current embedding:\n{}\n{}".format(results_table[0,0:3], new_record[0,0:3])
                results_table = np.append(results_table, new_record, axis=0)
                input.pickle_object(RESULTS_PATH, 'w', results_table)
                input.pickle_object(PLOT_RESULTS_PATH, 'w', results_table)
    
def main(arg=None):
    run_training(model_name)
    
if __name__=="__main__": 
    #tf.app.run()
    main()    
    
