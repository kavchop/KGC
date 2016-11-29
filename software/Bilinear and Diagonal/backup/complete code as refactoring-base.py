'''
Author of this implementation: Kavita Chopra (2016, version 1.0)

Source of conceptual model:

Bilinear model using Alternative Least Square Optimization (RESCAL): 
- "A three way model for collective learning on multi-relational data", 
   Nickel et al., 2011
  "Factorization of Yago" - Nickel et al., 2012
Tensor Network Formulation of Rescal: 
-  "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", 
    B. Yang et al. (2014)

Description of the Representation Learning Model
- Rescal is based on the factorization of a 3 mode tensor (3 dimensional matrix)
  where the aim is to solve the model parameters A and Mr given the low rank factorization 
  Xk = A x Mr x A^T
- A is a n x n_red (with rank n_red << n (#entities)) matrix representing the entity embedding
- Mr is a r x n_red matrix and there are as many different Mr's as there are relations in the triple store

The Neural Tensor Network Approach
- In a neural tensor network approach, this factorization is solved through a 2 layer network: 
- 1st layer: yi = xi * W  where xi is a one-hot vector for an entity representation (n-dimensional)
                      and W is a n x r matrix parameter that is to be learned
- 2nd layer: score = yh * Mr * yt where yh is the learned entity representation for a head
                                  Mr is the relation matrix for relation r and yt is the 
                                  learned entity representation for the tail entitiy 
- in the 2nd layer scores are computed for the positive and negative triple and optimized 
  by the loss function in a way, that positive triples get a higher score and negative triples a lower 


Variation of Rescal: Bilinear Diagonal Model
- In the bilinear diagonal model Mr is restricted to be a diagonal matrix to reduce the 
  complexity of the model 
- see "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", 
  B. Yang et al. (2014)
- run diagonal-bilinear training with "python bilinear.py --diagonal"
                                  


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
- run code with command line "python bilinear.py" for training with non-diagonal Mr or 
  "python bilinear.py --diagonal" to train the bilinear diagonal model 
- before training a new model:
    - meta-data-file with the customized configurations is created and saved in 'models/' directory
    - initial embedding is saved to disk for visualization purposes e.g. after dimensionality reduction through PCA
- at customizable intervals current model is saved to disk so that training may be continued in different sessions
- global- and hyper-parameters of the model can be configured in the params.py file. 

'''

import numpy as np
import tensorflow as tf
from datetime import datetime
import timeit 
import pickle 
import os
import zipfile 

os.chdir(os.getcwd())
DATA_PATH = "../../data/"

flags = tf.flags 
flags.DEFINE_bool("diagonal", False, "Train with diagonal relation matrices Mr")
FLAGS = flags.FLAGS

if FLAGS.diagonal: 
    model_name = 'diagonal'
    print "\nBilinear model is trained with diagonal relation matrices.\n"  
else:
    model_name = 'bilinear' 
    print "\nBilinear model is trained with non-diagonal relation matrices.\n"

    
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

#create vector files using int matrix and entity-to-vector map
def create_triple_array_batches(matrix_batch, ent_array_map):
    h_batch =  np.asarray([ent_array_map[matrix_batch[i,0]] for i in range(len(matrix_batch))])
    t_batch =  np.asarray([ent_array_map[matrix_batch[i,2]] for i in range(len(matrix_batch))])
    return h_batch, t_batch


#create map of triple batches by relation 
#during training iteration over such map to process the corresponding Mr for every mini-batch 
def create_triples_map_by_rel(int_matrix, m):
    triples_map = [np.reshape(np.array([], dtype=np.int32), (0,3)) for i in range(m)]

    for i in range(len(int_matrix)): 
        new_record =  np.reshape(int_matrix[i], (1,3))
        triples_map[int_matrix[i,1]]= np.append(triples_map[int_matrix[i,1]], new_record, axis=0)  
    return triples_map


# creates corrupted int matrix given positive training matrix 
def create_corrupt_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision): 
    if corrupt_two: 
        corrupt_batch_head = corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision, corrupt_head=True)
        corrupt_batch_tail = corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision, corrupt_head=False)
        corrupt_batch = np.append(corrupt_batch_head , corrupt_batch_tail , axis=0)
        return corrupt_batch
    else: 
        return corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision)


#corrupt randomly head of positive matrix or return matrix where only head or tail is corrupted
def corrupt_triple_matrix(triples_set, corrupt_two, matrix_batch, n, check_collision, corrupt_head=None): 
    if corrupt_head:
        cor_ind = 0  #corruption index: head
    else: 
        cor_ind = 2  # tail
        
    corrupt_batch = np.copy(matrix_batch) 
    if check_collision: 
        for i in range(len(corrupt_batch)):
            while tuple(corrupt_batch[i]) in triples_set:
                a = np.random.randint(n)  #n is number of all unique entities 
                if not corrupt_two:       #if random corruption  
                    cor_ind =  np.random.choice((0,2))
                corrupt_batch[i][cor_ind]= a
    else: 
        for i in range(len(corrupt_batch)):
            a = np.random.randint(n)  #n is number of all unique entities 
            if not corrupt_two: 
                cor_ind =  np.random.choice((0,2))
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
        
#normalize n x n_red matrix row-wise
def normalize_W(W):
    return np.array([W[i]/np.linalg.norm(W[i]) for i in range(len(W))])

#initialize model parameters W and Mr
def init_params(m, n, n_red): 
    if FLAGS.diagonal:
        Mr = np.random.rand(m, n_red,)
    else: 
        Mr = np.random.rand(m, n_red, n_red)
    W = np.random.rand(n,n_red)
    #normalize entitiy matrix
    W = normalize_W(W)
    return W, Mr

# diagonal matrix as 1 hot map for initial entity embedding of dim=n 
def create_1_hot_maps(n): 
    return np.eye(n)

#create matrix for learned entity embedding of dim=n_red: y = x*W where x is 1-hot vector 
def learned_ent_embed(ent_array_map, W_param):
    entity_embed = np.array([np.dot(ent_array_map[i], W_param) for i in range(len(ent_array_map))])
    return entity_embed

#method writes model configurations to disk 
def save_model_meta(model_name, n_red, learning_rate, corrupt_two, normalize_ent, check_collision):
    text_file = open("models/"+model_name+"_model_meta.txt", "w")
    text_file.write("\nmodel: {}\n\n".format(model_name))

    text_file.write("created on: {}\n".format(datetime.now().strftime('%d-%m-%Y %H:%M:%S')))
    text_file.write("factorization rank (dim.):  {}\n".format(n_red))
    text_file.write("learning rate:  {}\n".format(learning_rate))
    text_file.write("two corrupted per positive:  {}\n".format(corrupt_two))
    text_file.write("normalized entity matrix:  {}\n".format(normalize_ent))
    text_file.write("collision check:  {}\n".format(check_collision))
    text_file.close()

###################

#now methods for building the tensor network 

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
    
    if FLAGS.diagonal:
        layer_2_pos = tf.matmul((tf.mul(x_0_pos, Mr)), tf.transpose(x_1_pos))   #[none, none] = [none, n_red] * [n_red, n_red] * [n_red, none]
        layer_2_neg = tf.matmul((tf.mul(x_0_neg, Mr)), tf.transpose(x_1_neg))
    else: 
        layer_2_pos = tf.matmul((tf.matmul(x_0_pos, Mr)), tf.transpose(x_1_pos))   #[none, none] = [none, n_red] * [n_red, n_red] * [n_red, none]
        layer_2_neg = tf.matmul((tf.matmul(x_0_neg, Mr)), tf.transpose(x_1_neg))
        
    return layer_2_pos, layer_2_neg


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

#calculates the score for a triple batch of head, label, tail 
def score_func(h, l, t):
    score = np.dot(np.dot(h,l), (np.transpose(t)))
    return score  

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
def run_evaluation(triples_set, test_matrix, ent_array_map, Mr_param, 
                   test_size=None, eval_mode=False, verbose=False, l1_flag_test=True): 
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
        label = Mr_param[test_matrix[i,1]]
        if FLAGS.diagonal: 
            label = np.diag(label)
        correct_head = ent_array_map[test_matrix[i,0]] #save head before iterating 
        correct_tail = ent_array_map[test_matrix[i,2]] #save tail before iterating 
        correct_score_h = score_func(correct_head, label, tail)  #compute dissimilarity for correct triple
        correct_score_t = score_func(head, label,correct_tail)
        score_h_fixed = np.dot(head,label) 
        score_t_fixed = np.dot(label,(np.transpose(tail)))
        for j in range(n):       #iterate over all entities
            score_h = np.dot(ent_array_map[j], score_t_fixed) 
            score_t = np.dot(score_h_fixed, np.transpose(ent_array_map[j]))
            a_h[j] = score_h              #add dissimilarity to map a with entity string id from entity_list
            a_t[j] = score_t 
    
        #gives a map of entity to rank 
        #c is needed to get the rank of the correct entity: c[test_triples[i,0]]
        c_h = {key: rank for rank, key in enumerate(sorted(a_h, key=a_h.get, reverse=True), 1)}  
        c_t = {key: rank for rank, key in enumerate(sorted(a_t, key=a_t.get, reverse=True), 1)}
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

    n_red = 20                #dimension to which original dim n (number of entities) is reduced
    shuffle_data = False      #shuffle training data before every epoch to avoid any bias-effects in learning
    check_collision = False   #when creating corrupt training triples, check if triples is truly non-existent in whole data set 
    swap = True               #swap colums of raw triples files to bring to format subject predicate object 
    max_epoch = 50            #maximal epoch number for training 
    global_epoch = 0            #current epoch num, if training is continued on existing model, update with stored epoch_num 
    learning_rate = 0.1         #for Adagrad optimizer                
    batch_size = 100            #training batch size in every iteration of the SGD, depends on fixed batch size of training data from disk 
    result_log_cycle = 5       #run validation and log results after every x epochs
    embedding_log_cycle = 5    #save embedding to disk after every x epochs 
    corrupt_two = False         #for each pos triple create two corrupted triples 
    valid_size = 1000           #to speed up validation validate on a randomly drawn set of x from validation set
    valid_verbose = False       #display scores for each test triple during validation 
    train_verbose = False       #display loss after each gradient step
    normalize_ent = False
    
    #load set of all triples, train, valid and test data
    triples, train, valid, test = load_data(swap)
    ent_URI_to_int, rel_URI_to_int = create_dicts(triples)   #load dicts 

    
    #load int-matrices and triples-set for faster search
    triples_set, train_matrix, valid_matrix, test_matrix  = create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations
    
    #create 1 hot matrix for initial entity representation  
    ent_array_map = create_1_hot_maps(n)

    #create map of triple batches by relations 
    #during training iteration over such map to process the corresponding Mr for every mini-batch 
    triples_map = create_triples_map_by_rel(train_matrix, m)

    #load existing model or initialize new and save to disk
    if os.path.isfile(MODEL_PATH):
        print "\nExisting model is being loaded to continue training...\n"
        bilinear_model = pickle_object(MODEL_PATH, 'r')
        W_param = bilinear_model[0]
        Mr_param = bilinear_model[1]
    else: 
        #write model configurations to disk
        print "\nNew model is being initialized and saved before training starts...\n"
        save_model_meta(model_name, n_red, learning_rate, corrupt_two, normalize_ent, check_collision)
        W_param, Mr_param = init_params(m, n, n_red)
        bilinear_model = [W_param, Mr_param]
        pickle_object(INITIAL_MODEL, 'w', bilinear_model)
    
    entity_embed = learned_ent_embed(ent_array_map, W_param)

    #open eval-results table to retrieve the last trained global epoch
    #if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
    else: 
        results_table = np.reshape(np.asarray(['epoch', 'h_mean', 't_mean', 'h_hits', 't_hits'], dtype=object), (1,5))
        #run evaluation after initialization to get the state before training (at global_epoch 0)
        record = run_evaluation(triples_set, valid_matrix, entity_embed, Mr_param, 
                                test_size=valid_size)  
        new_record = np.reshape(np.asarray([global_epoch]+record), (1,5))
        print "validation result of current embedding: {}\n".format(new_record)
        results_table = np.append(results_table, new_record, axis=0)
        pickle_object(RESULTS_PATH, 'w', results_table)


    #Building the tensorflow graph input: 
    
    #entity representation matrix W, placeholder for W (W_ph) and assignment op W_op to 
    #load W with a learned W from a previous training session 
    W = tf.Variable(tf.zeros([n, n_red]))  
    W_ph = tf.placeholder(tf.float32, shape=(n, n_red))   
    W_op = tf.assign(W, W_ph)

    #relation matrix M, placeholder for M and assignment op M_op to load M with the current 
    #Mr corresponding to relation r 
    if FLAGS.diagonal: 
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

    #run evaluation before training of model
    #run_evaluation(triples_set, valid_matrix, entity_embed, Mr_param, eval_mode=False, verbose=True, test_size=valid_size) 

    
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
                    
                    h_batch, t_batch = create_triple_array_batches(pos_matrix, ent_array_map)
                    
                    if corrupt_two:
                        h_batch = np.append(h_batch, h_batch, axis=0)
                        t_batch = np.append(t_batch, t_batch, axis=0)

                    neg_matrix = create_corrupt_matrix(triples_set, corrupt_two, pos_matrix, n, check_collision)
                    h_1_batch, t_1_batch = create_triple_array_batches(neg_matrix, ent_array_map)
                        
                    feed_dict={x_0_pos: h_batch, x_1_pos: t_batch, 
                           x_0_neg: h_1_batch, x_1_neg: t_1_batch}
                    _, loss_value = sess.run(([trainer, loss]),
                                             feed_dict=feed_dict) 
                    
                    if train_verbose: 
                        print loss_value
                        
                    if normalize_ent: 
                        sess.run(W_norm)
                        
                    #update iterators l and r 
                    l += batch_size
                    if k == batch_num-2:
                        r = b_length
                    else: 
                        r += batch_size
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
                pickle_object(MODEL_PATH, 'w', bilinear_model)
                stop_embed = timeit.default_timer()
                print "done!\ntime taken to write to disk: {} sec\n".format((stop_embed-start_embed))
            if global_epoch == 1 or global_epoch%result_log_cycle == 0:  #may change j to num-epoch 
                entity_embed = learned_ent_embed(ent_array_map, W_param)
                record = run_evaluation(triples_set, valid_matrix, entity_embed, Mr_param, 
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
    
