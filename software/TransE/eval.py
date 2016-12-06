import numpy as np
import timeit
from input import pickle_object
import tensorflow as tf 




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


#L2 norm for TF tensors
def tensor_norm(tensor, l1_flag=True):
    if l1_flag:
        return tf.reduce_sum(tf.abs(tensor), reduction_indices=1)
    else: 
        return tf.sqrt(tf.reduce_sum(tf.square(tensor), reduction_indices=1))


#first helper methods for validation and evaluation:

def hits_at_ten(i, triples_set, test_matrix, a_h, a_t, abs_hits_h, abs_hits_t, top_triples):
    #create list of tuples of entity and rank sorted by rank (descendingly) 
    b_h = sorted(a_h.items(), key=lambda x: x[1], reverse=False)
    b_t = sorted(a_t.items(), key=lambda x: x[1], reverse=False)
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

def print_verbose_results(i, entity_offset, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode, rel_hits_h=None, rel_hits_t=None):
    if eval_mode: 
        print "\ntest triple entity {}:\nmean rank from head repl.: {}  -  more scores:".format(i+entity_offset, c_h[test_matrix[i,0]])
        print "correct: {}    min: {}     max: {}     hit: {}".format(correct_score, 
                                                                             min(a_h.values()), max(a_h.values()),
                                                                             rel_hits_h)
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "correct: {}    min: {}     max: {}     hit: {}".format(correct_score, 
                                                                 min(a_t.values()), max(a_t.values()),
                                                                 rel_hits_t)
    else: 
        print "\ntest triple entity {}:\nmean rank from head repl.: {}  -  more scores:".format(i+entity_offset, c_h[test_matrix[i,0]])
        print "correct: {}    min: {}     max: {}".format(correct_score, 
                                                                             min(a_h.values()), max(a_h.values()))
                                                                    
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "correct: {}    min: {}     max: {}:".format(correct_score, 
                                                                 min(a_t.values()), max(a_t.values()))
                                                                 
def print_final_results(rank_mean_h, rank_mean_t, n, eval_mode, rel_hits_h=None, rel_hits_t=None):
    print "\nrank_mean resulting from head replacement: {} (out of total rank number {})".format(rank_mean_h, n)
    print "rank_mean resulting from tail replacement: {} (out of total rank number {})".format(rank_mean_t, n)
    if eval_mode: 
        print "hits@ten for head repl.: {}%".format(rel_hits_h * 100)
        print "hits@ten for tail repl.: {}%".format(rel_hits_t * 100)



#now follows main method for validation/evaluation: run_evaluation()

def run_evaluation(triples_set, total_test_matrix, ent_array_map, rel_array_map, 
                   score_func, test_size=None, eval_mode=False, verbose=False, l1_flag=True): 
  
    n = len(ent_array_map)
    dim = len(ent_array_map[0])


    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        # Graph input

        #placeholders and assignment ops to feed the graph with the currrent input batches
        h_ph = tf.placeholder(tf.float32, shape=(None, dim))     #head (subject)
        t_ph = tf.placeholder(tf.float32, shape=(None, dim))     #tail (object)
        l_ph = tf.placeholder(tf.float32, shape=(None, dim))     #label (relation)
        
        h_1 = tf.placeholder(tf.float32, shape=(None, dim))      #head from corrupted counterpart triple 
        t_1 = tf.placeholder(tf.float32, shape=(None, dim))      #tail from corrupted counterpart triple 

        tensor_op = tensor_norm(h_ph+l_ph-t_ph, l1_flag)
 


	if test_size != None: 
		selected_indices = np.random.randint(len(total_test_matrix), size=test_size)
		total_test_matrix = total_test_matrix[selected_indices] 

        CONST = 5
        sub_matrix_list = np.split(total_test_matrix, CONST)
    

        #op for Variable initialization 
        #init_op = tf.initialize_all_variables() 
	init_op = tf.global_variables_initializer()
        
        sess.run(init_op)

        start = timeit.default_timer()  #stop = timeit.default_timer()
        print "\nValidation of current embedding starts!"

        result_sum = np.zeros(4) 
        print "\nintermediate results from sub-batches: "
        for c in range(CONST):
	        rank_sum_h = 0
	        rank_sum_t = 0
	        hit_ten_h = []
	        hit_ten_t = []
	        abs_hits_h = 0    #absolute number of hits from top ten ranked triples
	        abs_hits_t = 0
	        top_triples =  np.reshape(np.array([]), (0,3))

	        a_h_map = {}
	        a_t_map = {}
	        a_h = {}
	        a_t = {}
	        for i in range(len(ent_array_map)):  
		 	a_h_map[i] = {}
		 	a_t_map[i] = {}

	        t_batch_map = {}
	        h_batch_map = {}
	        score_vector_t, score_vector_h = {}, {}

                test_matrix = sub_matrix_list[c]
		h_batch =  np.asarray([ent_array_map[test_matrix[i,0]] for i in range(len(test_matrix))])
		l_batch =  np.asarray([rel_array_map[test_matrix[i,1]] for i in range(len(test_matrix))])

		for i in range(len(ent_array_map)):
			t_batch_map[i] =  np.asarray([ent_array_map[i] for j in range(len(test_matrix))])
		
		#for iteration over head (fixed tail and label) 
		t_batch =  np.asarray([ent_array_map[test_matrix[i,2]] for i in range(len(test_matrix))])
		h_batch_map = t_batch_map.copy()
		
		for i in range(len(ent_array_map)):
			score_vector_t[i] = sess.run(tensor_op, feed_dict = {h_ph: h_batch, l_ph: l_batch, t_ph: t_batch_map[i]})
		        score_vector_h[i] = sess.run(tensor_op, feed_dict = {h_ph: h_batch_map[i], l_ph: l_batch, t_ph: t_batch})
		       
		for i in range(len(test_matrix)): 
	       		for k in range(len(ent_array_map)): 
	 			a_h_map[i][k] = score_vector_h[k][i]
		                a_t_map[i][k] = score_vector_t[k][i]
	 
		        a_h = a_h_map[i]
		        a_t = a_t_map[i]
		        correct_score = float(sess.run(tensor_op, feed_dict = {h_ph:np.reshape(ent_array_map[test_matrix[i,0]], (1,dim)), l_ph: np.reshape(rel_array_map[test_matrix[i,1]], (1,dim)), t_ph: np.reshape(ent_array_map[test_matrix[i,2]],(1,dim))}))
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
			    print_verbose_results(i, entity_offset, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode, rel_hits_h, rel_hits_t)
		
			if not eval_mode and verbose: #case not eval and verbose
			    print_verbose_results(i, entity_offset,  test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode)
			#else (if not verbose), no need to print 
		
		rank_mean_h = rank_sum_h/len(test_matrix) 
		rank_mean_t = rank_sum_t/len(test_matrix)
	
		if eval_mode: 
			print_final_results(rank_mean_h, rank_mean_t, n, eval_mode, rel_hits_h, rel_hits_t)
			pickle_object('top_triples', 'w', top_triples)
			result_sum += np.array([rank_mean_h, rank_mean_t, rel_hits_h, rel_hits_t])
		else:
 			if c%2 == 0:
				print_final_results(rank_mean_h, rank_mean_t, n, eval_mode)
			result_sum += np.array([rank_mean_h, rank_mean_t, 0, 0])
    	
        stop = timeit.default_timer()
        print "\ntime taken for validation: {} min".format((stop - start)/ 60) 
        final_result = list(np.array(result_sum/CONST, dtype=np.int32))
        print final_result
        return final_result 
