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
DATA_PATH = "../../data/"

diagonal = False
model_name ='bilinear'

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
  


#methods for building the tensor network 

#layer 1 for dim-reduction for all entities in current batch: y = x * W
def Layer_1(x_0_pos, x_1_pos, x_0_neg, x_1_neg, W):
    layer_1_pos_0 = tf.nn.relu(tf.matmul(x_0_pos, W))  #[none, n_red] = [none, n] * [n, n_red]
    layer_1_pos_1 = tf.nn.relu(tf.matmul(x_1_pos, W))
    layer_1_neg_0 = tf.nn.relu(tf.matmul(x_0_neg, W))
    layer_1_neg_1 = tf.nn.relu(tf.matmul(x_1_neg, W))
    return layer_1_pos_0, layer_1_pos_1, layer_1_neg_0, layer_1_neg_1

#layer 2 calculates the pos and neg score used for loss function
#score: x0 * Mr * x1
def Layer_2(x_0_pos, x_1_pos, x_0_neg, x_1_neg, Mr): 
    
    if diagonal:
        layer_2_pos = tf.matmul((tf.mul(x_0_pos, Mr)), tf.transpose(x_1_pos))   #[none, none] = [none, n_red] * [n_red, n_red] * [n_red, none]
        layer_2_neg = tf.matmul((tf.mul(x_0_neg, Mr)), tf.transpose(x_1_neg))
    else: 
        layer_2_pos = tf.matmul((tf.matmul(x_0_pos, Mr)), tf.transpose(x_1_pos))   #[none, none] = [none, n_red] * [n_red, n_red] * [n_red, none]
        layer_2_neg = tf.matmul((tf.matmul(x_0_neg, Mr)), tf.transpose(x_1_neg))
        
    return layer_2_pos, layer_2_neg



def run_training(model_name): 
    
    #set global- and hyper-parameters, for description of each param see file params.py

    n_red = params.n_red
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
    normalize_ent = params.normalize_ent
 

    if eval_mode: 
	mode = 'evaluated'
    else: 
        mode = 'trained'

    if model_name == 'diagonal':
	print "\nBilinear model is {} with diagonal relation matrices.\n".format(mode)  
    else:
    	print "\nBilinear model is {} with non-diagonal relation matrices.\n".format(mode)
    
    if not normalize_ent: 
        model_name = model_name + ' not normalized'


    MODEL_PATH='models/'+model_name +'_model'
    INITIAL_MODEL='models/'+model_name+'_initial_model'
    RESULTS_PATH='models/'+model_name +'_results'


    #load set of all triples, train, valid and test data
    triples, train, valid, test = input.load_data(swap)
    ent_URI_to_int, rel_URI_to_int = input.create_dicts(triples)   #load dicts 

    
    #load int-matrices and triples-set for faster search
    triples_set, train_matrix, valid_matrix, test_matrix  = input.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations

    
    #create 1 hot matrix for initial entity representation  
    ent_array_map = input.create_1_hot_maps(n)

    #create map of triple batches by relations 
    #during training iteration over such map to process the corresponding Mr for every mini-batch 
    triples_map = input.create_triples_map_by_rel(train_matrix, m)


    #load existing model or initialize new and save to disk
    if os.path.isfile(MODEL_PATH):
        print "\nExisting model is being loaded...\n"
        bilinear_model = input.pickle_object(MODEL_PATH, 'r')
        W_param = bilinear_model[0]
        Mr_param = bilinear_model[1]
        entity_embed = input.learned_ent_embed(ent_array_map, W_param)

	#run evaluation before training of model
	if eval_mode:	 
		eval.run_evaluation(diagonal, triples_set, valid_matrix, entity_embed, Mr_param, eval_mode=True, verbose=True, test_size=valid_size) 
		return
    else: 
        if eval_mode: 
		print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(model_name)
		return	
        #write model configurations to disk
        print "\nNew model is being initialized and saved before training starts...\n"
        input.save_model_meta(model_name, n_red, learning_rate, corrupt_two, normalize_ent, check_collision)
        W_param, Mr_param = input.init_params(m, n, n_red)
        bilinear_model = [W_param, Mr_param]
        input.pickle_object(INITIAL_MODEL, 'w', bilinear_model)
    
    entity_embed = input.learned_ent_embed(ent_array_map, W_param)

    #open eval-results table to retrieve the last trained global epoch
    #if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = input.pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
	new_lines = "\ntraining resumed on {}\nat epoch: {}\nwith learning rate: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), global_epoch, learning_rate)
	with open('models/transe_model_meta.txt', "a") as f:
		f.write(new_lines)
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits'], dtype=object), (1,5))
        #run evaluation after initialization to get the state before training (at global_epoch 0)
        record = eval.run_evaluation(diagonal, triples_set, valid_matrix, entity_embed, Mr_param, test_size=valid_size)  
        new_record = np.reshape(np.asarray([global_epoch]+record), (1,5))
        print "validation result of current embedding: {}\n".format(new_record)
        results_table = np.append(results_table, new_record, axis=0)
        input.pickle_object(RESULTS_PATH, 'w', results_table)


    #Building the tensorflow graph input: 
    
    #entity representation matrix W, placeholder for W (W_ph) and assignment op W_op to 
    #load W with a learned W from a previous training session 
    W = tf.Variable(tf.zeros([n, n_red]))  
    W_ph = tf.placeholder(tf.float32, shape=(n, n_red))   
    W_op = tf.assign(W, W_ph)

    #relation matrix M, placeholder for M and assignment op M_op to load M with the current 
    #Mr corresponding to relation r 
    if diagonal: 
        M = tf.Variable(tf.zeros([n_red,]))
        M_ph = tf.placeholder(tf.float32, shape=(n_red,))
        M_op = tf.assign(M, M_ph)
    else: 
        M = tf.Variable(tf.zeros([n_red, n_red]))
        M_ph = tf.placeholder(tf.float32, shape=(n_red, n_red))
        M_op = tf.assign(M, M_ph)
    
    #input mini-batches fed at runtime during training 
    x_0_pos = tf.placeholder(tf.float32, shape=(None, n))
    x_1_pos = tf.placeholder(tf.float32, shape=(None, n))
    x_0_neg = tf.placeholder(tf.float32, shape=(None, n))
    x_1_neg = tf.placeholder(tf.float32, shape=(None, n))
    
    
    #Building the graph 
    x_1, x_2, x_3, x_4 = Layer_1(x_0_pos, x_1_pos, x_0_neg, x_1_neg, W)
    pos_score, neg_score = Layer_2(x_1, x_2, x_3, x_4, M)
   
    loss = tf.reduce_sum(tf.maximum(tf.sub(neg_score, pos_score)+1, 0))

    trainer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    #alternatively: less complex AdamOptimizer: 
    #trainer=tf.train.AdamOptimizer().minimize(loss)

    #ops for normalizing W
    normed = tf.sqrt(tf.reduce_sum(tf.square(W), 1, keep_dims=True))
    W_new = tf.div(W,normed)
    W_norm = tf.assign(W, W_new)                
    
    #op for Variable initialization 
    init_op = tf.initialize_all_variables()
     

    
    print "\nNumber of Triples in Training data: {}".format(len(train_matrix))
    print "Iteration over training batches by relations (# {}) and maximal batch-size of {}".format(m, batch_size)

        
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(W_op, feed_dict = {W_ph: W_param})
        print "\nTraining starts!\n"
        for i in range(max_epoch):
            print "epoch: {}".format(i)
            start = timeit.default_timer()
            for j in range(m):
                #load M variable with current Mr  
                sess.run(M_op, feed_dict = {M_ph: Mr_param[j]}) 
                b_length = len(triples_map[j]) #number of triples with common relation j
                #print b_length
                X_id = np.arange(b_length)
                if shuffle_data: 
                    #vector X_id mirrors indices training batch to allow 
                    #inexpensive shuffling before each epoch
                    np.random.shuffle(X_id)
                #initialize batch iteration indices  
                batch_num = b_length/batch_size
                if b_length%batch_size != 0: 
                    batch_num +=1
                if b_length < batch_size:
                    batch_num = 1
                    r = b_length
                else: 
                    r = batch_size
                l = 0
                #now iterate over triples with common relations 
                #and learn from sub-batches of maximally batch_size
                for k in range(batch_num): 
                    
                    pos_matrix = triples_map[j][X_id[l:r]]
                    
                    h_batch, t_batch = input.create_triple_array_batches(pos_matrix, ent_array_map)
                    
                    if corrupt_two:
                        h_batch = np.append(h_batch, h_batch, axis=0)
                        t_batch = np.append(t_batch, t_batch, axis=0)

                    neg_matrix = input.create_corrupt_matrix(triples_set, corrupt_two, pos_matrix, n, check_collision)
                    h_1_batch, t_1_batch = input.create_triple_array_batches(neg_matrix, ent_array_map)
                        
                    feed_dict={x_0_pos: h_batch, x_1_pos: t_batch, 
                           x_0_neg: h_1_batch, x_1_neg: t_1_batch}
                    _, loss_value = sess.run(([trainer, loss]),
                                             feed_dict=feed_dict) 
                    
                    if train_verbose: 
                        print loss_value
                        
                    #update iterators l and r 
                    l += batch_size
                    if k == batch_num-2:
                        r = b_length
                    else: 
                        r += batch_size
		if normalize_ent: 
                        sess.run(W_norm)
                #after processing all subbatches with common relation save updated Mr 
                Mr_param[j] = M.eval()
            stop = timeit.default_timer()
            print "time taken for current epoch: {} min".format((stop - start)/60)
            global_epoch += 1
            if global_epoch%embedding_log_cycle == 0:
                W_param = W.eval()
                start_embed = timeit.default_timer()
                print "writing emebdding to disk..."
                bilinear_model = [W_param, Mr_param]
                input.pickle_object(MODEL_PATH, 'w', bilinear_model)
                stop_embed = timeit.default_timer()
                print "done!\ntime taken to write to disk: {} sec\n".format((stop_embed-start_embed))
            if global_epoch == 1 or global_epoch%result_log_cycle == 0:  #may change j to num-epoch 
                entity_embed = input.learned_ent_embed(ent_array_map, W_param)
                record = eval.run_evaluation(diagonal, triples_set, valid_matrix, entity_embed, Mr_param, test_size=valid_size, verbose=valid_verbose) 
                new_record = np.reshape(np.asarray([global_epoch]+record), (1,5))
                #TODO: LINES FOR EARLY STOPPING
    
                print "validation result of current embedding: {}".format(new_record)
                results_table = np.append(results_table, new_record, axis=0)
                input.pickle_object(RESULTS_PATH, 'w', results_table)
    
def main(arg=None):
    run_training(model_name)
    
if __name__=="__main__": 
    #tf.app.run()
    main()    
    
