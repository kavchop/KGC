'''
Author of code: Kavita Chopra (10.2016, version 1.0)

Source of conceptual model:
- Translating Embeddings for Modeling Multi-relational Data
  (A. Bordes et al.), 2013

Brief Description of the Representation Learning Model: 
- TransE embeds entities and relations from a knowledge base into a vector space
- Linear model that starting from a random initial embedding of dimension k 
  (hyperparameter) learns embeddings in a way that true triples get a a higher score 
  and false triples a lower score

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
    - initial embedding is saved to disk for visualization purposes e.g. after dimensionality reduction through PCA
- at customizable intervals current model is saved to disk so that training may be continued in different sessions
- global- and hyper-parameters of the model can be configured in the params.py file. 

'''


import numpy as np
import tensorflow as tf
import pickle 
from math import sqrt
import timeit
from datetime import datetime
import os
import zipfile 



model_name = 'transe'

os.chdir(os.getcwd())
DATA_PATH = "../../data/"
    
MODEL_PATH='models/'+model_name +'_model'
INITIAL_MODEL='models/'+model_name+'_initial_model'
RESULTS_PATH='models/'+model_name +'_results'

###################

#helper methods creation of input files  

#creates int-to-URI dicts for entities and relations 
#input is the complete triple-store 
def parse_triple_store(triples):  
    #else create new dicts
    e1 = triples[:,0]   #subject
    e2 = triples[:,2]   #object 
    rel = triples[:,1]  #predicate
    ent_int_to_URI = list(set(e1).union(set(e2)))
    rel_int_to_URI = list(set(rel))
    return ent_int_to_URI, rel_int_to_URI

#only URI-to-int dicts are saved to disk
#int_to_URI dicts are needed to create the former
def create_URI_to_int(ent_int_to_URI, rel_int_to_URI):
    ent_URI_to_int = {ent_int_to_URI[i]: i for i in range(len(ent_int_to_URI))}    
    rel_URI_to_int = {rel_int_to_URI[i]: i for i in range(len(rel_int_to_URI))}
    return ent_URI_to_int, rel_URI_to_int

#method for preprocessing triple files such that columns are swapped to format sub-pred-obj
def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]] 

#create integer matrix from triple-store usings dicts
def create_matrix(triples, ent_URI_to_int, rel_URI_to_int): 
    int_matrix =  np.asarray([[ent_URI_to_int[triples[i,0]], rel_URI_to_int[triples[i,1]], ent_URI_to_int[triples[i,2]]] for i in range(len(triples))])
    return int_matrix



def create_triple_array_batches(matrix_batch, ent_array_map, rel_array_map, corrupt=False):
    h_batch =  np.asarray([ent_array_map[matrix_batch[i,0]] for i in range(len(matrix_batch))])
    t_batch =  np.asarray([ent_array_map[matrix_batch[i,2]] for i in range(len(matrix_batch))])

    if corrupt: 
        return h_batch, t_batch
    l_batch =  np.asarray([rel_array_map[matrix_batch[i,1]] for i in range(len(matrix_batch))])
    return h_batch, l_batch, t_batch


def corrupt_triple_matrix(triples_set, matrix_batch, n, check_collision=True): 
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
def load_data(swap=None):
    #if any of the triple files is missing unpack data.zip
    if not os.path.isfile(DATA_PATH +'train.txt') or not os.path.isfile(DATA_PATH +'test.txt') or not os.path.isfile(DATA_PATH +'valid.txt'): 
        zip_ref = zipfile.ZipFile(DATA_PATH +'data.zip', 'r')
        zip_ref.extractall(DATA_PATH)
        zip_ref.close()
    
    train_triples = np.loadtxt(DATA_PATH +'train.txt',dtype=np.object,comments='#',delimiter=None)
    test_triples = np.loadtxt(DATA_PATH +'test.txt',dtype=np.object,comments='#',delimiter=None) #59071 triples
    valid_triples = np.loadtxt(DATA_PATH +'valid.txt',dtype=np.object,comments='#',delimiter=None) #50000 triples
    
    if swap: 
        swap_cols(train_triples, 1,2)
        swap_cols(test_triples, 1,2)
        swap_cols(valid_triples, 1,2)
    triples = np.concatenate((train_triples, test_triples, valid_triples), axis=0)
    return triples, train_triples, valid_triples, test_triples


#create URI_to_int dicts for entitiy and relations 
def create_dicts(triples):
    if os.path.isfile(DATA_PATH +'URI_to_int'): 
        URI_to_int = pickle_object(DATA_PATH +'URI_to_int', 'r')
        ent_URI_to_int = URI_to_int[0]
        rel_URI_to_int = URI_to_int[1]
        return ent_URI_to_int, rel_URI_to_int
    else: 
        ent_int_to_URI, rel_int_to_URI = parse_triple_store(triples)
        ent_URI_to_int, rel_URI_to_int = create_URI_to_int(ent_int_to_URI, rel_int_to_URI)
        URI_to_int = [ent_URI_to_int, rel_URI_to_int]
        pickle_object(DATA_PATH + 'URI_to_int', 'w', URI_to_int)
        return ent_URI_to_int, rel_URI_to_int

#given URI triple matrices and URI-to-int dicts create int triple matrices and
#set of triples for faster search through triple store 
def create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int):
    triples_matrix = create_matrix(triples, ent_URI_to_int, rel_URI_to_int)
    train_matrix = create_matrix(train, ent_URI_to_int, rel_URI_to_int)
    valid_matrix = create_matrix(valid, ent_URI_to_int, rel_URI_to_int)
    test_matrix = create_matrix(test, ent_URI_to_int, rel_URI_to_int)
    
    triples_set = set((tuple(triples_matrix[i])) for i in range(len(triples_matrix)))
    return triples_set, train_matrix, valid_matrix, test_matrix 

#method for either reading from or writing to disk a pickle object
def pickle_object(FILE_NAME, mode, object_=None):
    if mode == 'w':
        file = open(FILE_NAME, 'w')
        pickle.dump(object_, file)
        file.close()
    if mode == 'r':
        file = open(FILE_NAME, 'r')
        object_ = pickle.load(file)
        file.close()
        return object_
        

def numpy_norm(x, l1_flag=True):
    if l1_flag:
        return np.linalg.norm((x), ord=1)
    else: 
        return np.linalg.norm(x)

def normalize_embedding(x, l1_flag=True):
    return np.array([x[i]/numpy_norm(x[i], l1_flag) for i in range(len(x))])

def normalize_vec(x, l1_flag=True):
    return x/numpy_norm(x, l1_flag)
    
def create_embed_maps_from_int(n,m, dim):
    ent_array_map = [np.random.uniform(-6/sqrt(dim),6/sqrt(dim), (dim,)) for i in range(n)]   
    rel_array_map = [np.random.uniform(-6/sqrt(dim),6/sqrt(dim), (dim,)) for i in range(m)]

    ent_array_map = normalize_embedding(ent_array_map)
    rel_array_map = normalize_embedding(rel_array_map)
    return ent_array_map, rel_array_map

#computes the distance score of a triple batch: x=(h+l-t)
#used in validation part 
def calc_dissimilarity(x, l1_flag=True):
    return numpy_norm(x, l1_flag)


#method writes model configurations to disk 
def save_model_meta(model_name, dim, learning_rate, normalize_ent, check_collision):
    text_file = open("models/"+model_name+"_model_meta.txt", "w")
    text_file.write("\nmodel: {}\n\n".format(model_name))

    text_file.write("created on: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
    text_file.write("dimension:  {}\n".format(dim))
    text_file.write("learning rate:  {}\n".format(learning_rate))
    text_file.write("normalized entity matrix:  {}\n".format(normalize_ent))
    text_file.write("collision check:  {}\n".format(check_collision))
    text_file.close()

#L2 norm for TF tensors
def tensor_norm(tensor, l1_flag=True):
    if l1_flag:
        return tf.reduce_sum(tf.abs(tensor))
    else: 
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

###################

'''
Validation Protocol:
-iterate over all test triples
-calculate the true score based on optimal embedding from training 
-fix tail and label (and then head and label)
-iterate over all possible entities from entity list and calculate scores
-order entities by scores 
-get the score rank for the correct entity 
-calculate the mean of all such ranks

Evaluation Protocol: 
-for every test triple evaluated in validation part, check for the entities 
 with the top 10 scores whether they are present in the complete triple store
-report the proportion of the true triples
'''

#first helper methods for validation and evaluation:

def hits_at_ten(i, triples_set, test_matrix, a_h, a_t, abs_hits_h, abs_hits_t, top_triples):
    #create list of tuples of entity and rank sorted by rank (descendingly) 
    b_h = sorted(a_h.items(), key=lambda x: x[1], reverse=True)
    b_t = sorted(a_t.items(), key=lambda x: x[1], reverse=True)
    for k in range(10):
        temp_h =  [b_h[k][0], test_matrix[i,1], test_matrix[i,2]]
        if tuple(temp_h) in triples_set:
            top_triples = np.append(top_triples, np.reshape(np.array(temp_h), (1,3)), axis=0)
            abs_hits_h += 1
        temp_t =  [test_matrix[i,0], test_matrix[i,1], b_t[k][0]]
        if tuple(temp_t) in triples_set:
            top_triples = np.append(top_triples, np.reshape(np.array(temp_t), (1,3)), axis=0)
            abs_hits_t += 1
            
    #compute relative hits from absolute hits 
    rel_hits_h = float(abs_hits_h)/(10*len(test_matrix)) 
    rel_hits_t = float(abs_hits_t)/(10*len(test_matrix)) 
    return abs_hits_h, abs_hits_t, rel_hits_h, rel_hits_t, top_triples

def print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score_h, correct_score_t, eval_mode, rel_hits_h=None, rel_hits_t=None):
    if eval_mode: 
        print "\ntest triple entity {}:\nmean rank from head repl.: {}  -  more scores:".format(i, c_h[test_matrix[i,0]])
        print "correct: {}    min: {}     max: {}     hit: {}".format(correct_score_h, 
                                                                             min(a_h.values()), max(a_h.values()),
                                                                             rel_hits_h)
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "correct: {}    min: {}     max: {}     hit: {}".format(correct_score_t, 
                                                                 min(a_t.values()), max(a_t.values()),
                                                                 rel_hits_t)
    else: 
        print "\ntest triple entity {}:\nmean rank from head repl.: {}  -  more scores:".format(i, c_h[test_matrix[i,0]])
        print "correct: {}    min: {}     max: {}".format(correct_score_h, 
                                                                             min(a_h.values()), max(a_h.values()))
                                                                    
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "correct: {}    min: {}     max: {}:".format(correct_score_t, 
                                                                 min(a_t.values()), max(a_t.values()))
                                                                 
def print_final_results(rank_mean_h, rank_mean_t, n, eval_mode, rel_hits_h=None, rel_hits_t=None):
    print "rank_mean resulting from head replacement: {} (out of total rank number {})".format(rank_mean_h, n)
    print "rank_mean resulting from tail replacement: {} (out of total rank number {})".format(rank_mean_t, n)
    if eval_mode: 
        print "hits@ten for head repl.: {}%".format(rel_hits_h * 100)
        print "hits@ten for tail repl.: {}%".format(rel_hits_t * 100)



#now follows main method for validation/evaluation: run_evaluation()

def run_evaluation(triples_set, test_matrix, ent_array_map, rel_array_map, 
                   score_func, 
                   test_size=None, eval_mode=False, verbose=False, l1_flag=True): 
    if test_size != None: 
        selected_indices = np.random.randint(len(test_matrix), size=test_size)
        test_matrix = test_matrix[selected_indices] 

    start = timeit.default_timer()
    print "\nValidation of current embedding starts!"
    n = len(ent_array_map)
    rank_sum_h = 0
    rank_sum_t = 0
    hit_ten_h = []
    hit_ten_t = []
    abs_hits_h = 0    #absolute number of hits from top ten ranked triples
    abs_hits_t = 0
    top_triples =  np.reshape(np.array([]), (0,3))
    
    for i in range(len(test_matrix)):
        a_h = {}
        a_t = {}
        tail = ent_array_map[test_matrix[i,2]]    # fix tail and label, iterate over head
        head = ent_array_map[test_matrix[i,0]]    # fix head and label, iterate over tail
        label = rel_array_map[test_matrix[i,1]]
        correct_head = ent_array_map[test_matrix[i,0]] #save head before iterating 
        correct_tail = ent_array_map[test_matrix[i,2]] #save tail before iterating 
        correct_score_h = score_func(correct_head + label - tail)  #compute dissimilarity for correct triple
        correct_score_t = score_func(head + label - correct_tail)
        #h + l -t 
        fixed_tail = label-tail
        fixed_head = head+label
        for j in range(n):       #iterate over all entities
            score_h = score_func(ent_array_map[j] + fixed_tail, l1_flag)    #compute dissimilarity 
            score_t = score_func(fixed_head - ent_array_map[j], l1_flag)
            a_h[j] = score_h              #add dissimilarity to map a with entity string id from entity_list
            a_t[j] = score_t 
    
        #gives a map of entity to rank 
        #c is needed to get the rank of the correct entity: c[test_triples[i,0]]
        c_h = {key: rank for rank, key in enumerate(sorted(a_h, key=a_h.get, reverse=False), 1)}  
        c_t = {key: rank for rank, key in enumerate(sorted(a_t, key=a_t.get, reverse=False), 1)}
        rank_sum_t = rank_sum_t + c_t[test_matrix[i,2]] #add rank of correct entity to current rank_sum 
        rank_sum_h = rank_sum_h + c_h[test_matrix[i,0]] #add rank of correct entity to current rank_sum

        #printing intermediate scores during eval/validation
        if eval_mode:
            abs_hits_h, abs_hits_t, rel_hits_h, rel_hits_t, top_triples = hits_at_ten(i, triples_set, test_matrix, a_h, a_t, abs_hits_h, abs_hits_t, top_triples) 
        
        if eval_mode and verbose: #case eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score_h, correct_score_t, eval_mode, rel_hits_h, rel_hits_t)
        
        if not eval_mode and verbose: #case not eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score_h, correct_score_t, eval_mode)
        #else (if not verbose), no need to print 
        
    rank_mean_h = rank_sum_h/len(test_matrix)  
    rank_mean_t = rank_sum_t/len(test_matrix) 
    stop = timeit.default_timer()
    print "\ntime taken for validation: {} min".format((stop - start)/ 60)
    #print final results: 
    
    if eval_mode: 
        print_final_results(rank_mean_h, rank_mean_t, n, eval_mode, rel_hits_h, rel_hits_t)
        pickle_object('top_triples', 'w', top_triples)
        return [rank_mean_h, rank_mean_t, rel_hits_h, rel_hits_t]
    else:
        print_final_results(rank_mean_h, rank_mean_t, n, eval_mode)
        return [rank_mean_h, rank_mean_t, 0, 0]
        


def run_training(): 
    
    #global- and hyper-parameters

    dim = 20                 #dimension of the embedding space for both entities and relations
    shuffle_data = False     #shuffle training data before every epoch to avoid any bias-effects in learning
    check_collision = False  #when creating corrupt training triples, check if triples is truly non-existent in whole data set 
    swap = True              #swap colums of raw triples files to bring to format subject predicate object 
    max_epoch = 50           #maximal epoch number for training 
    global_epoch = 0         #current epoch num, if training is continued on existing model, update with stored epoch_num 
    margin = 2               #param for loss function; greater than 0, e.g. {1, 2, 10}
    learning_rate = 0.1      #for Adagrad optimizer                
    batch_size = 100         #training batch size in every iteration of the SGD, depends on fixed batch size of training data from disk 
    embedding_log_cycle = 5     #save embedding to disk after every x epochs 
    l1_flag = True              #for normalization and distance measure in loss function 
    normalize_ent = False       #normalize entities after each mini-batch iteration
    result_log_cycle = 1        #run validation and log results after every x epochs
    valid_size = 1000           #to speed up validation validate on a randomly drawn set of x from validation set
    valid_verbose = False       #display scores for each test triple during validation 
    train_verbose = False       #display loss after each gradient step
    
     
    #load set of all triples, train, valid and test data
    triples, train, valid, test = load_data(swap)
    ent_URI_to_int, rel_URI_to_int = create_dicts(triples)   #load dicts 
    #load triples_set for faster existential checks, and int_matrices 
    triples_set, train_matrix, valid_matrix, test_matrix  = create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations

    #load existing model or initialize new and save to disk
    if os.path.isfile(MODEL_PATH):
        print "\nExisting TransE model is being loaded to continue training...\n"
        transE_model = pickle_object(MODEL_PATH, 'r')
        ent_array_map = transE_model[0]
        rel_array_map = transE_model[1]
    else: 
        #write model configurations to disk
        print "\nNew TransE model is being initialized and saved before training starts...\n"
        save_model_meta(model_name, dim, learning_rate, normalize_ent, check_collision)
        ent_array_map, rel_array_map = create_embed_maps_from_int(n,m,dim)
        transE_model = [ent_array_map,rel_array_map]
        pickle_object('models/transe_initial_model', 'w', transE_model)
        

    #open eval-results table to retrieve the last trained epoch
    #if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits']), (1,5))
        #run evaluation after initialization to get the state before training (at epoch 0)
        record = run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, 
                                score_func=calc_dissimilarity, test_size=valid_size)  
        new_record = np.reshape(np.asarray([global_epoch]+record), (1,5))
        print "validation result of current embedding: {}".format(new_record)
        results_table = np.append(results_table, new_record, axis=0)
        transE_model = pickle_object(RESULTS_PATH, 'w', results_table)


    #run evaluation before training of model
    run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, 
                   score_func = calc_dissimilarity, 
                   eval_mode=False, verbose=True, test_size=valid_size) 

    with tf.Session() as sess:
        # Graph input

        # TF variables that learn in a mini-batch iteration
        h = tf.Variable(tf.zeros([batch_size, dim]))     
        t = tf.Variable(tf.zeros([batch_size, dim]))
        l = tf.Variable(tf.zeros([batch_size, dim]))

        #placeholders and assignment ops to feed the graph with the currrent input batches
        h_ph = tf.placeholder(tf.float32, shape=(None, dim))     #head (subject)
        t_ph = tf.placeholder(tf.float32, shape=(None, dim))     #tail (object)
        l_ph = tf.placeholder(tf.float32, shape=(None, dim))     #label (relation)
        
        h_1 = tf.placeholder(tf.float32, shape=(None, dim))      #head from corrupted counterpart triple 
        t_1 = tf.placeholder(tf.float32, shape=(None, dim))      #tail from corrupted counterpart triple 

        h_op = tf.assign(h, h_ph)  
        t_op = tf.assign(t, t_ph)
        l_op = tf.assign(l, l_ph)

        #loss function 
        loss = tf.reduce_sum(margin + tensor_norm((h + l) - t, l1_flag) - tensor_norm((h_1 + l) - t_1, l1_flag))

        #building training:
        trainer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        #alternatively: less complex AdamOptimizer: 
        #trainer=tf.train.AdamOptimizer().minimize(loss)

        batch_num = len(train_matrix)/batch_size
    
        print "\nNumber of Triples in Training data:{}".format(len(train_matrix))
        print "With a batch_size of {} we have {} number of batches in an epoch".format(batch_size, batch_num)


        #vector X_id mirrors indices of train_matrix to allow inexpensive shuffling 
        #before each epoch
        X_id = np.arange(len(train_matrix))

        #op for Variable initialization 
        init_op = tf.initialize_all_variables()

        
        sess.run(init_op)
        print "\nTraining of TransE starts!"
        for i in range(max_epoch):
            print "\nepoch: {}".format(i)
            if shuffle_data: 
                np.random.shuffle(X_id)
            #init left and right index of current batch from training data 
            lt = 0
            rt = batch_size
            start = timeit.default_timer()
            for j in range(batch_num): 
                pos_matrix=np.copy(train_matrix[X_id[lt:rt]])
                h_batch, l_batch, t_batch = create_triple_array_batches(pos_matrix, ent_array_map, rel_array_map)
                neg_matrix = corrupt_triple_matrix(triples_set, pos_matrix, n)
                h_1_batch, t_1_batch = create_triple_array_batches(neg_matrix, ent_array_map, rel_array_map, corrupt=True)
                #run assign ops, to transfer current batch input as tensorflow variables 
                sess.run(h_op, feed_dict = {h_ph: h_batch})
                sess.run(t_op, feed_dict = {t_ph: t_batch})
                sess.run(l_op, feed_dict = {l_ph: l_batch})
                #feed the non-variables of the training data,
                #that is h_1 and t_1 from negative set:
                feed_dict={h_1: h_1_batch, t_1: t_1_batch} 
                _, loss_value = sess.run(([trainer, loss]), feed_dict=feed_dict)
                if train_verbose:
                    print loss_value
                #extract the learned variables from this iteration as numpy arrays
                numpy_h = h.eval()
                numpy_t = t.eval()
                numpy_l = l.eval()
                #save learned variables in global entitiy array map 
                for k in range(len(pos_matrix)):
                    ent_array_map[pos_matrix[k,0]] = numpy_h[k]    
                    ent_array_map[pos_matrix[k,2]] = numpy_t[k]
                    rel_array_map[pos_matrix[k,1]] = numpy_l[k]
                    if normalize_ent: 
                        ent_array_map[pos_matrix[k,0]] = normalize_vec(numpy_h[k])   
                        ent_array_map[pos_matrix[k,2]] = normalize_vec(numpy_t[k])
                #update l and r: 
                lt += batch_size
                rt += batch_size

            stop = timeit.default_timer()
            print "time taken for current epoch: {} min".format((stop - start)/60)
            global_epoch += 1
            #save model after each embedding_log_cycle
            if global_epoch%embedding_log_cycle == 0:
                start_embed = timeit.default_timer()
                print "writing emebdding to disk..."
                transE_model = [ent_array_map, rel_array_map]
                pickle_object(MODEL_PATH, 'w', transE_model)
                stop_embed = timeit.default_timer()
                print "done!\ntime taken to write to disk: {} sec\n".format((stop_embed-start_embed))
            if global_epoch == 1 or global_epoch%result_log_cycle == 0:  #may change j to num-epoch 
                record = run_evaluation(triples_set, valid_matrix, ent_array_map, rel_array_map, 
                                        score_func = calc_dissimilarity, 
                                        test_size=valid_size, verbose=valid_verbose) 
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
    
