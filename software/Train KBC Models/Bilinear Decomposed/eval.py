import numpy as np
from datetime import datetime
import timeit 
import pickle 



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

#helper methods for validation and evaluation:

#calculates the score for a triple batch of head, label, tail 
def score_func(h, l, t):
    score = np.dot(np.dot(h,l), (np.transpose(t)))
    return score  

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
    #check if current triple has a score in top 10 ranked triples and add to top_triples if so
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
            
    #compute relative hits from absolute hits 
    rel_hits_h = float(abs_hits_h)/(len(test_matrix)) 
    rel_hits_t = float(abs_hits_t)/(len(test_matrix)) 
    return abs_hits_h, abs_hits_t, rel_hits_h, rel_hits_t, top_triples


def print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode, rel_hits_h=None, rel_hits_t=None):
    if eval_mode: 
        print "\ntest triple entity {}:\nmean rank from head repl.: {}  -  more scores:".format(i, c_h[test_matrix[i,0]])
        print "correct: {}    min: {}     max: {}     hit: {}".format(correct_score, 
                                                                             min(a_h.values()), max(a_h.values()),
                                                                             rel_hits_h)
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "correct: {}    min: {}     max: {}     hit: {}".format(correct_score, 
                                                                 min(a_t.values()), max(a_t.values()),
                                                                 rel_hits_t)
    else: 
        print "\ntest triple entity {}:\nmean rank from head repl.: {}  -  more scores:".format(i, c_h[test_matrix[i,0]])
        print "correct: {}    min: {}     max: {}".format(correct_score, 
                                                                             min(a_h.values()), max(a_h.values()))
                                                                    
        print "mean rank from tail repl.: {}  -  more scores:".format(c_t[test_matrix[i,2]])
        print "correct: {}    min: {}     max: {}:".format(correct_score, 
                                                                 min(a_t.values()), max(a_t.values()))
                                                                 
def print_final_results(rank_mean_h, rank_mean_t, n, eval_mode, rel_hits_h=None, rel_hits_t=None):
    print "rank_mean resulting from head replacement: {} (out of total rank number {})".format(rank_mean_h, n)
    print "rank_mean resulting from tail replacement: {} (out of total rank number {})".format(rank_mean_t, n)
    if eval_mode: 
        print "hits@ten for head repl.: {}%".format(rel_hits_h * 100)
        print "hits@ten for tail repl.: {}%".format(rel_hits_t * 100)


#now follows main method for validation/evaluation: run_evaluation()
#def run_evaluation(diagonal, triples_set, test_matrix, ent_array_map, Mr_param, test_size=None, eval_mode=False, verbose=False, l1_flag_test=True): 

def run_evaluation(triples_set, test_matrix, ent_array_map, Mr_param,
                   score_func,
                   test_size=None, eval_mode=False, filtered=False, verbose=False, l1_flag=True): 


    diagonal = False
    # for faster test runs allow validation on smaller (random) subset of test_matrix by specifiying a test_size for the set 
    np.random.seed(seed = 20) 
    if test_size != None: 
        selected_indices = np.random.randint(len(test_matrix), size=test_size)
        test_matrix = test_matrix[selected_indices] 
    # if not specified, set test_size to total size of test_matrix 
    if test_size == None: 
	test_size = len(test_matrix)  

    start = timeit.default_timer()
    print "\nValidation of current embedding starts!"
    n = len(ent_array_map)

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

    top_triples =  np.reshape(np.array([]), (0,3))
   
    for i in range(len(test_matrix)):
        a_h = {}
        a_t = {}
        tail = ent_array_map[test_matrix[i,2]]    # fix tail and label, iterate over head
        head = ent_array_map[test_matrix[i,0]]    # fix head and label, iterate over tail
        label = Mr_param[test_matrix[i,1]]
        if diagonal: 
            label = np.diag(label)
        #correct_score = score_func(head, label, tail)  #compute dissimilarity for correct triple
	correct_score = 0
        score_h_fixed = np.dot(head,label) 
        score_t_fixed = np.dot(label,(np.transpose(tail)))
        for j in range(n):       #iterate over all entities
            if filtered: 
		    temp =  [j, test_matrix[i,1], test_matrix[i,2]]
		    if tuple(temp) not in triples_set or j==test_matrix[i,0]:
			    score_h = np.dot(ent_array_map[j], score_t_fixed) 
			    a_h[j] = score_h
		    temp =  [test_matrix[i,0], test_matrix[i,1], j]
		    if tuple(temp) not in triples_set or j==test_matrix[i,2]:
		    	    score_t = np.dot(score_h_fixed, np.transpose(ent_array_map[j]))
			    a_t[j] = score_t
	    else: 
		    score_h = np.dot(ent_array_map[j], score_t_fixed) 
            	    score_t = np.dot(score_h_fixed, np.transpose(ent_array_map[j]))
                    a_h[j] = score_h              #add dissimilarity to map a with entity string id from entity_list
                    a_t[j] = score_t 
	
                   
        #c is needed to get the rank of the correct entity: c[test_triples[i,0]]
        c_h = {key: rank for rank, key in enumerate(sorted(a_h, key=a_h.get, reverse=True), 1)}  
        c_t = {key: rank for rank, key in enumerate(sorted(a_t, key=a_t.get, reverse=True), 1)}
        map_sum_t = map_sum_t + c_t[test_matrix[i,2]] #add rank of correct entity to current rank_sum 
        map_sum_h = map_sum_h + c_h[test_matrix[i,0]] #add rank of correct entity to current rank_sum

        #mean reciprocal rank:
        mrr_sum_t = mrr_sum_t + 1.0/c_t[test_matrix[i,2]] 
	mrr_sum_h = mrr_sum_h + 1.0/c_h[test_matrix[i,0]]	
        #printing intermediate scores during eval/validation
        if eval_mode:
            abs_hits_h, abs_hits_t, rel_hits_h, rel_hits_t, top_triples = hits_at_ten(i, c_h, c_t, triples_set, test_matrix, a_h, a_t, abs_hits_h, abs_hits_t, top_triples) 
        
        if eval_mode and verbose: #case eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode, rel_hits_h, rel_hits_t)
        
        if not eval_mode and verbose: #case not eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode)
        #else (if not verbose), no need to print 
        
    # mean average precision:    
    map_h = map_sum_h/len(test_matrix)
    map_t = map_sum_t/len(test_matrix)
    # mean reciprocal rank: 
    mrr_h = mrr_sum_h/len(test_matrix)  
    mrr_t = mrr_sum_t/len(test_matrix)
  
    stop = timeit.default_timer()
    print "\ntime taken for validation: {} min\n".format((stop - start)/ 60)

    # MRR were a later added extension but MAP remain the focus of the validation/ evaluation 
    print "Mean Reciprocal Ranks"
    print 'MRR from head replacement: ', mrr_h
    print 'MRR from tail replacement: ', mrr_t
    if filtered: 
	print "\nNote that results are filtered\n"    
    else: 
    	print "\nNote that results are unfiltered\n" 

    # print final results: 
    if eval_mode: 
        print_final_results(map_h, map_t, n, eval_mode, rel_hits_h, rel_hits_t)
        return top_triples
    else: 
        print_final_results(map_h, map_t, n, eval_mode)
        return [map_h, map_t, 0, 0]
        


