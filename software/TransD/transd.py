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
from math import sqrt
import timeit
from datetime import datetime
import os
import sys
import zipfile 
import params
import input
from input import pickle_object
import eval

model_name = 'transd'

MODEL_PATH='models/'+model_name +'_model'
INITIAL_MODEL='models/'+model_name+'_initial_model'
RESULTS_PATH='models/'+model_name +'_results'
MODEL_META_PATH='models/'+model_name +'_meta.txt'

eval_mode = False

if len(sys.argv)==2:
	if sys.argv[1] == 'evaluate': 
	    eval_mode = True


#L2 norm for TF tensors
def tensor_norm(tensor, l1_flag=True):
    if l1_flag:
        return tf.reduce_sum(tf.abs(tensor))
    else: 
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))


def run_training(): 
    
    #set global- and hyper-parameters, for description of each param see file params.py

    dim_ent = params.dim_ent
    dim_rel = params.dim_rel
    shuffle_data = params.shuffle_data 
    check_collision = params.check_collision
    swap = params.swap
    max_epoch = params.max_epoch
    global_epoch = params.global_epoch
    margin = params.margin
    learning_rate = params.learning_rate              
    batch_size = params.batch_size
    l1_flag = params.l1_flag
    result_log_cycle = params.result_log_cycle
    embedding_log_cycle = params.embedding_log_cycle
    valid_size = params.valid_size
    valid_verbose = params.valid_verbose
    train_verbose = params.train_verbose
    normalize_ent = params.normalize_ent
    
     
    #load set of all triples, train, valid and test data
    triples, train, valid, test = input.load_data(swap)
    ent_URI_to_int, rel_URI_to_int = input.create_dicts(triples)   #load dicts 
    #load triples_set for faster existential checks, and int_matrices 
    triples_set, train_matrix, valid_matrix, test_matrix  = input.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations
    #load existing model or initialize new and save to disk
    if os.path.isfile(MODEL_PATH):
        print "\nExisting model is being loaded...\n"
        trans_model = pickle_object(MODEL_PATH, 'r')
        ent_array_map = trans_model[0]
        rel_array_map = trans_model[1]
        ent_p_array_map = trans_model[2]
        rel_p_array_map = trans_model[3]

	#run evaluation before training of model
	if eval_mode:	 
		eval.run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, ent_p_array_map, rel_p_array_map, score_func = input.calc_dissimilarity, eval_mode=True, verbose=True, test_size=valid_size) 
		return
    else: 
        if eval_mode: 
		print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(model_name)
		return
        #write model configurations to disk
        print "\nNew model is being initialized and saved before training starts...\n"
        input.save_model_meta(model_name, dim_ent, dim_rel, learning_rate, normalize_ent, check_collision)
        ent_array_map, rel_array_map , ent_p_array_map, rel_p_array_map= input.create_embed_maps_from_int(n,m,dim_ent, dim_rel)
        trans_model = [ent_array_map,rel_array_map, ent_p_array_map, rel_p_array_map]
        pickle_object(INITIAL_MODEL, 'w', trans_model)
        

    #open eval-results table to retrieve the last trained epoch
    #if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
        new_lines = "\ntraining resumed on {}\nat epoch: {}\nwith learning rate: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), global_epoch, learning_rate)
	with open(MODEL_META_PATH, "a") as f:
		f.write(new_lines)
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits']), (1,5))
        #run evaluation after initialization to get the state before training (at epoch 0)
        record = eval.run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, ent_p_array_map, rel_p_array_map, score_func=input.calc_dissimilarity, test_size=valid_size)  
        new_record = np.reshape(np.asarray([global_epoch]+record), (1,5))
        print "validation result of current embedding: {}".format(new_record)
        results_table = np.append(results_table, new_record, axis=0)
        trans_model = pickle_object(RESULTS_PATH, 'w', results_table)


    with tf.Session() as sess:
        # Graph input

        # TF variables that learn in a mini-batch iteration
        h = tf.Variable(tf.zeros([batch_size, dim_ent]))     
        t = tf.Variable(tf.zeros([batch_size, dim_ent]))
        l = tf.Variable(tf.zeros([batch_size, dim_rel]))
        
        hp = tf.Variable(tf.zeros([batch_size, dim_ent]))     
        tp = tf.Variable(tf.zeros([batch_size, dim_ent]))
        lp = tf.Variable(tf.zeros([batch_size, dim_rel]))
        

        #placeholders and assignment ops to feed the graph with the currrent input batches
        h_ph = tf.placeholder(tf.float32, shape=(None, dim_ent))     #head (subject)
        t_ph = tf.placeholder(tf.float32, shape=(None, dim_ent))     #tail (object)
        l_ph = tf.placeholder(tf.float32, shape=(None, dim_rel))     #label (relation)
        
        hp_ph = tf.placeholder(tf.float32, shape=(None, dim_ent))     #head (subject)
        tp_ph = tf.placeholder(tf.float32, shape=(None, dim_ent))     #tail (object)
        lp_ph = tf.placeholder(tf.float32, shape=(None, dim_rel))     #label (relation)
        
        h_1 = tf.placeholder(tf.float32, shape=(None, dim_ent))      #head from corrupted counterpart triple 
        t_1 = tf.placeholder(tf.float32, shape=(None, dim_ent))      #tail from corrupted counterpart triple 
        
        hp_1 = tf.placeholder(tf.float32, shape=(None, dim_ent))      #head from corrupted counterpart triple 
        tp_1 = tf.placeholder(tf.float32, shape=(None, dim_ent))      #tail from corrupted counterpart triple 

        h_op = tf.assign(h, h_ph)  
        t_op = tf.assign(t, t_ph)
        l_op = tf.assign(l, l_ph)
        
        hp_op = tf.assign(hp, hp_ph)  
        tp_op = tf.assign(tp, tp_ph)
        lp_op = tf.assign(lp, lp_ph)
        
        I = tf.Variable(initial_value = np.eye(dim_rel, dim_ent), dtype=tf.float32)
        
        Mrh = tf.matmul(tf.transpose(lp), hp) + I
        Mrt = tf.matmul(tf.transpose(lp), tp) + I
        
        Mrh_1 = tf.matmul(tf.transpose(lp), hp_1) + I
        Mrt_1 = tf.matmul(tf.transpose(lp), tp_1) + I

        h_ = tf.matmul(h, tf.transpose(Mrh))
        t_ = tf.matmul(t, tf.transpose(Mrt))
        
        h_1_ = tf.matmul(h_1, tf.transpose(Mrh_1))
        t_1_ = tf.matmul(t_1, tf.transpose(Mrt_1))
        
        #loss function 
        loss = tf.reduce_sum(margin + tensor_norm((h_ + l) - t_, l1_flag) - tensor_norm((h_1_ + l) - t_1_, l1_flag))

        #building training:
        trainer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        #alternatively: less complex AdamOptimizer: 
        #trainer=tf.train.AdamOptimizer().minimize(loss)

        batch_num = len(train_matrix)/batch_size
    
        print "\nNumber of Triples in Training data: {}".format(len(train_matrix))
        print "With a batch_size of {} we have {} number of batches in an epoch".format(batch_size, batch_num)


        #vector X_id mirrors indices of train_matrix to allow inexpensive shuffling 
        #before each epoch
        X_id = np.arange(len(train_matrix))

        #op for Variable initialization 
        init_op = tf.initialize_all_variables()

        
        sess.run(init_op)
        print "\nTraining starts!"
        for i in range(max_epoch):
            print "\nepoch: {}".format(i)
            if shuffle_data: 
                np.random.shuffle(X_id)
            #init left and right index of current batch from training data 
            lt = 0
            rt = batch_size
            start = timeit.default_timer()
            for j in range(batch_num): 
                pos_matrix = np.copy(train_matrix[X_id[lt:rt]])
                h_batch, l_batch, t_batch, hp_batch, lp_batch, tp_batch = input.create_triple_array_batches(pos_matrix, ent_array_map, rel_array_map, ent_p_array_map, rel_p_array_map)
                neg_matrix = input.corrupt_triple_matrix(triples_set, pos_matrix, n)
                h_1_batch, t_1_batch, hp_1_batch, tp_1_batch = input.create_triple_array_batches(neg_matrix, ent_array_map, rel_array_map, ent_p_array_map, rel_p_array_map, corrupt=True)
        
                #run assign ops, to transfer current batch input as tensorflow variables 
                sess.run(h_op, feed_dict = {h_ph: h_batch})
                sess.run(t_op, feed_dict = {t_ph: t_batch})
                sess.run(l_op, feed_dict = {l_ph: l_batch})
                
                sess.run(hp_op, feed_dict = {hp_ph: hp_batch})
                sess.run(tp_op, feed_dict = {tp_ph: tp_batch})
                sess.run(lp_op, feed_dict = {lp_ph: lp_batch})
                #feed the non-variables of the training data, 
                #that is h_1 and t_1 from negative set:
                feed_dict={h_1: h_1_batch, t_1: t_1_batch, hp_1: hp_1_batch, tp_1: tp_1_batch} 
                _, loss_value = sess.run(([trainer, loss]), feed_dict=feed_dict)
                if train_verbose:
                    print loss_value
                #extract the learned variables from this iteration as numpy arrays
                numpy_h = h.eval()
                numpy_t = t.eval()
                numpy_l = l.eval()
                numpy_hp = hp.eval()
                numpy_tp = tp.eval()
                numpy_lp = lp.eval()
                #save learned variables in global entitiy array map 
                for k in range(len(pos_matrix)):
                    ent_array_map[pos_matrix[k,0]] = numpy_h[k]    
                    ent_array_map[pos_matrix[k,2]] = numpy_t[k]
                    rel_array_map[pos_matrix[k,1]] = numpy_l[k]
                    ent_p_array_map[pos_matrix[k,0]] = numpy_hp[k]    
                    ent_p_array_map[pos_matrix[k,2]] = numpy_tp[k]
                    rel_p_array_map[pos_matrix[k,1]] = numpy_lp[k]
                #update l and r: 
                lt += batch_size
                rt += batch_size
	    if normalize_ent: 
		for k in range(n):
                    ent_array_map[k] = input.normalize_vec(ent_array_map[k])   
                    ent_p_array_map[k] = input.normalize_vec(ent_p_array_map[k]) 
 
            stop = timeit.default_timer()
            print "time taken for current epoch: {} min".format((stop - start)/60)
            global_epoch += 1
            #save model after each embedding_log_cycle
            if global_epoch%embedding_log_cycle == 0:
                start_embed = timeit.default_timer()
                print "writing emebdding to disk..."
                trans_model = [ent_array_map, rel_array_map, ent_p_array_map, rel_p_array_map]
                pickle_object(MODEL_PATH, 'w', trans_model)
                stop_embed = timeit.default_timer()
                print "done!\ntime taken to write to disk: {} sec\n".format((stop_embed-start_embed))
            if global_epoch == 1 or global_epoch%result_log_cycle == 0:  #may change j to num-epoch 
                record = eval.run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, ent_p_array_map, rel_p_array_map, score_func = input.calc_dissimilarity, test_size=valid_size, verbose=valid_verbose) 
                new_record = np.reshape(np.asarray([global_epoch]+record), (1,5))
                #TODO: LINES FOR EARLY STOPPING
                
                print "validation result of current embedding: {}".format(new_record)
                results_table = np.append(results_table, new_record, axis=0)
                pickle_object(RESULTS_PATH, 'w', results_table)

    
def main(arg=None):
    run_training()
    
if __name__=="__main__": 
    tf.app.run()
    #main()    
    
