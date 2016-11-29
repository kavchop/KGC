import numpy as np
import timeit
from input import pickle_object




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

def run_evaluation(triples_set, test_matrix, ent_array_map, rel_array_map, 
                   ent_p_array_map, rel_p_array_map, score_func, 
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
    dim_ent = ent_array_map.shape[1]
    dim_rel = rel_array_map.shape[1]
    ent_array_map = np.reshape(ent_array_map, (len(ent_array_map), 1, dim_ent))
    ent_p_array_map = np.reshape(ent_p_array_map, (len(ent_array_map), 1, dim_ent))
    rel_array_map = np.reshape(rel_array_map, (len(rel_array_map), 1, dim_rel))
    rel_p_array_map = np.reshape(rel_p_array_map, (len(rel_array_map), 1, dim_rel))

    I = np.zeros((dim_rel,dim_ent), int)
    np.fill_diagonal(I, 1)  

    for i in range(len(test_matrix)):
        a_h = {}
        a_t = {}
	label = rel_array_map[test_matrix[i,1]]
        label_p = rel_p_array_map[test_matrix[i,1]]
        tail = ent_array_map[test_matrix[i,2]]
	head = ent_array_map[test_matrix[i,0]]
	#print label_p.shape, ent_p_array_map[test_matrix[i,2]].shape, I.shape
        tail_p = np.dot(np.dot(np.transpose(label_p), ent_p_array_map[test_matrix[i,2]]) + I, np.transpose(tail))   
        head_p = np.dot(np.dot(np.transpose(label_p), ent_p_array_map[test_matrix[i,0]]) + I, np.transpose(head)) 
        correct_score = score_func(head_p + label - tail_p, l1_flag)  #compute dissimilarity for correct triple
        #h + l -t 
        fixed_tail = label-tail_p  # fix tail and label, iterate over head
        fixed_head = head_p+label  # fix head and label, iterate over tail
        for j in range(n):       #iterate over all entities
            score_h = score_func(np.dot(np.dot(np.transpose(label_p), ent_p_array_map[j]) + I, np.transpose(ent_array_map[j])) + fixed_tail, l1_flag)    #compute dissimilarity 
            score_t = score_func(fixed_head - np.dot(np.dot(np.transpose(label_p), ent_p_array_map[j]) + I, np.transpose(ent_array_map[j])), l1_flag)
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
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score,eval_mode, rel_hits_h, rel_hits_t)
        
        if not eval_mode and verbose: #case not eval and verbose
            print_verbose_results(i, test_matrix, a_h, a_t, c_h, c_t, correct_score, eval_mode)
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