import numpy as np
import timeit
from input import pickle_object
import os
import input




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

# first helper methods for validation and evaluation:
 
def hits_at_ten(i, c_h, c_t, triples_set, test_matrix, a_h, a_t, abs_hits_h, abs_hits_t, top_triples):
    '''
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
    '''
    # check if current triple has a score in top 10 ranked triples and add to top_triples if so
    flag_in_set = False
    if c_h[test_matrix[i,0]] in set(np.arange(11)):
    	abs_hits_h += 1
        flag_in_set = True
    if c_t[test_matrix[i,2]] in set(np.arange(11)):
	abs_hits_t += 1
        flag_in_set = True

    temp = [test_matrix[i,0], test_matrix[i,1], test_matrix[i,2]]
    if flag_in_set:
	top_triples = np.append(top_triples, np.reshape(np.array(temp), (1,3)), axis=0)
         
    # compute relative hits from absolute hits 
    rel_hits_h = float(abs_hits_h)/(len(test_matrix)) 
    rel_hits_t = float(abs_hits_t)/(len(test_matrix)) 
    return abs_hits_h, abs_hits_t, rel_hits_h, rel_hits_t, top_triples


def print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode, rel_hits_h=None, rel_hits_t=None):

    print "\n\n----- test triple entity {}: -----\n".format(i)

    if eval_mode:
        print "mean rank from head repl.: {}".format(c_h[test_matrix[i,0]])
        print "  min: {}     max: {}     correct: {}".format(min(a_h.values()), max(a_h.values()), correct_score)
        print "  current h_hit: {}".format(rel_hits_h)
        print "mean rank from tail repl.: {}".format(c_t[test_matrix[i,2]])
        print "  min: {}     max: {}     correct: {}".format(min(a_t.values()), max(a_t.values()), correct_score)
        print "  current t_hit: {}".format(rel_hits_t)

    else: 
        print "mean rank from head repl.: {}".format(c_h[test_matrix[i,0]])
        print "  min: {}     max: {}     correct: {}".format(min(a_h.values()), max(a_h.values()), correct_score)
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "  min: {}     max: {}     correct: {}:".format(min(a_t.values()), max(a_t.values()), correct_score)
                                                                 
def print_final_results(rank_mean_h, rank_mean_t, n, eval_mode, rel_hits_h=None, rel_hits_t=None):
    print "\nMean Average Precision: "
    print "rank_mean resulting from head replacement: {} (out of total rank number {})".format(rank_mean_h, n)
    print "rank_mean resulting from tail replacement: {} (out of total rank number {})".format(rank_mean_t, n)
    if eval_mode: 
        print "hits@ten for head repl.: {}%".format(rel_hits_h * 100)
        print "hits@ten for tail repl.: {}%".format(rel_hits_t * 100)



# now follows main method for validation/evaluation: run_evaluation()

def run_evaluation(triples_set, test_matrix, ent_array_map, rel_array_map, 
                   score_func, test_size=None, eval_mode=False, filtered=False, verbose=False): 

    # for faster test runs allow validation on smaller (random) subset of test_matrix by specifiying a test_size for the set 
    np.random.seed(seed = 20) 
    if test_size != None: 
        selected_indices = np.random.randint(len(test_matrix), size=test_size)
        test_matrix = test_matrix[selected_indices] 
    # if not specified, set test_size to total size of test_matrix 
    if test_size == None: 
	test_size = len(test_matrix)  

    # number of entities
    n = len(ent_array_map)
    dim = ent_array_map.shape[1]

    # for mean average precision
    map_sum_h = 0
    map_sum_t = 0
    # for mean reciprocal ranks
    mrr_sum_h = 0
    mrr_sum_t = 0
    # collects the counts for the true triples with a rank <= 10 (hits @ ten)
    hit_ten_h = []
    hit_ten_t = []
    # absolute number of hits from top ten ranked triples
    abs_hits_h = 0   
    abs_hits_t = 0

    print "\n\n******Validation of current embedding starts!******"
    start = timeit.default_timer()

    # during evaluation (not validation) all true triples with map in top 10 ranks will be stored in top_triples for later visualization etc. 
    top_triples =  np.reshape(np.array([]), (0,3))
    
    # entitiy list from 0 - n
    entity_list_h = np.reshape(np.arange(0,n), (n,1))
    entity_list_t = np.reshape(np.arange(0,n), (n,1))
    n_h, n_t = n, n

    # iteration over all test triples
    for i in range(test_size): 

        # for filtered results, compute and rank the scores only for triples which are truly false, that is, they are not contained in the entire triple set; based on this information determine the list of entities to test against:  
	if filtered:
                n = len(ent_array_map)
		set_h = set(np.arange(0,n))
		set_t = set(np.arange(0,n))
                #print 'set'
		#print len(set_h), len(set_t)
	        for j in range(n):
			temp =  [j, test_matrix[i,1], test_matrix[i,2]]
		    	if tuple(temp) in triples_set and not j==test_matrix[i,0]:
				set_h.remove(j)	
		        temp =  [test_matrix[i,0], test_matrix[i,1], j]
		    	if tuple(temp) in triples_set and not j==test_matrix[i,2]:
				set_t.remove(j)	
                #print len(set_h), len(set_t)
		entity_list_h = np.reshape(np.array(list(set_h)), (len(set_h), 1))
        	entity_list_t = np.reshape(np.array(list(set_t)), (len(set_t), 1))
                n_h = len(entity_list_h)
                n_t = len(entity_list_t)

        # maps which collect the scores for every replaced entity replacements 
        a_h = {}
        a_t = {}

        # create the label vector from the embedding and calculate the correct score
	#label = np.repeat(np.reshape(rel_array_map[test_matrix[i,1]], (1,dim)), n, axis=0)
        # correct_score = input.score_func(ent_array_map[test_matrix[i,0]], rel_array_map[test_matrix[i,1]], ent_array_map[test_matrix[i,2]], l1_flag)
	correct_score = 0

        # iteration over head (fixed tail):
        test_triple_h = np.reshape(test_matrix[i,1:3], (1,2))                     # (l, t)
        temp = np.reshape(np.repeat(test_triple_h, n_h, axis=0), (n_h,2))             #(l,t) x n 
        triple_matrix_h = np.concatenate([entity_list_h, temp], axis=1)  #(hi, l xn, t xn)

	head = ent_array_map[triple_matrix_h[:,0]] 
        # since tail is fixed for whole ith batch, take embedding and repeat n x times 
        if i == n-1:
        	print i
        	print 'n_h', n_h
        tail = ent_array_map[triple_matrix_h[:,2]] 
	#tail = np.repeat(np.reshape(ent_array_map[triple_matrix_h[i,2]], (1,dim)), n_h, axis=0)
        #label = np.repeat(np.reshape(rel_array_map[test_matrix[i,1]], (1,dim)), n_h, axis=0)  
	label = rel_array_map[triple_matrix_h[:,1]] 

	# compute list of scores: 
	scores = -1 * np.linalg.norm((head + label - tail), ord=1, axis= 1)
        # map from entity to score for all test test triples where head had been substitued  
        a_h = dict(zip(entity_list_h.flatten(), scores.tolist()))

	# iteration over tail (fixed head):
        test_triple_t = np.reshape(test_matrix[i,0:2], (1,2))                     # (h, l)
        temp = np.reshape(np.repeat(test_triple_t, n_t, axis=0), (n_t,2))             #(h,l) x n 
        triple_matrix_t = np.concatenate([temp, entity_list_t], axis=1)  #(h xn, l xn, ti)
     
	# since head is fixed for whole ith batch, take embedding and repeat n x times 
        head = ent_array_map[triple_matrix_t[:,0]] 
	#head = np.repeat(np.reshape(ent_array_map[triple_matrix_t[i,0]], (1,dim)), n_t, axis=0)  
	tail = ent_array_map[triple_matrix_t[:,2]] 
        if filtered: 
        	label = np.repeat(np.reshape(rel_array_map[test_matrix[i,1]], (1,dim)), n_t, axis=0)
        
        # compute list of scores:
        scores = -1 * np.linalg.norm((head + label - tail), ord=1, axis= 1)
        # map from entity to score for all test test triples where tail had been substitued  
        a_t = dict(zip(entity_list_t.flatten(), scores.tolist())) 

        # gives a map of entity to rank 
        # c is needed to get the rank of the correct entity: c[test_triples[i,0]]
        c_h = {key: rank for rank, key in enumerate(sorted(a_h, key=a_h.get, reverse=True), 1)}  
        c_t = {key: rank for rank, key in enumerate(sorted(a_t, key=a_t.get, reverse=True), 1)}

	# mean average precision: 
        map_sum_t = map_sum_t + c_t[test_matrix[i,2]] #add rank of correct entity to current rank_sum 
        map_sum_h = map_sum_h + c_h[test_matrix[i,0]] #add rank of correct entity to current rank_sum
                   
        # update mean reciprocal rank - sums
        mrr_sum_t = mrr_sum_t + 1.0/c_t[test_matrix[i,2]] 
	mrr_sum_h = mrr_sum_h + 1.0/c_h[test_matrix[i,0]]		

        # printing intermediate scores during eval/validation
        if eval_mode:
            abs_hits_h, abs_hits_t, rel_hits_h, rel_hits_t, top_triples = hits_at_ten(i, c_h, c_t, triples_set, test_matrix, a_h, a_t, abs_hits_h, abs_hits_t, top_triples) 
        
        if eval_mode and verbose: #case eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode, rel_hits_h, rel_hits_t)
        
        if not eval_mode and verbose: #case not eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode)
        # else (if not verbose), no need to print 
    
    # mean average precision:    
    map_h = map_sum_h/len(test_matrix)
    map_t = map_sum_t/len(test_matrix)
    # mean reciprocal rank: 
    mrr_h = mrr_sum_h/len(test_matrix)  
    mrr_t = mrr_sum_t/len(test_matrix)

    if filtered: 
	print "\nFiltered results: \n"    
    else: 
    	print "\nUnfiltered results: \n" 

    # MRR were a later added extension but MAP remain the focus of the validation/ evaluation 
    print "Mean Reciprocal Ranks"
    print 'MRR from head replacement: ', mrr_h
    print 'MRR from tail replacement: ', mrr_t

    # print final results: 
    if eval_mode: 
        print_final_results(map_h, map_t, n, eval_mode, rel_hits_h, rel_hits_t)
        stop = timeit.default_timer()
    	print "\ntime taken for validation: {} min\n".format((stop - start)/ 60)
        return top_triples
    else: 
        print_final_results(map_h, map_t, n, eval_mode)
        stop = timeit.default_timer()
        print "\ntime taken for validation: {} min\n".format((stop - start)/ 60)
        return [map_h, map_t, 0, 0]

