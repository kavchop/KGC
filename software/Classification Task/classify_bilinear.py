
import numpy as np
import timeit
import os
#from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import roc
#from sklearn import metrics
from KBC_Util_Class import KBC_Util_Class


model_name = 'Bilinear'
dim = 20

dataset = 'Freebase'
swap = True



#computes the distance score of a triple batch: x=(h+l-t)
def score_func(h, l, t):
    score = np.dot(np.dot(h,l), (np.transpose(t)))
    return int(score)


# method takes positive and negative matrix of triples, computes and return scores and a target vector of 1s and 0s
def get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map):
	    h_batch = ent_array_map[pos_matrix[:,0]]
            l_batch = rel_array_map[pos_matrix[:,1]]
	    t_batch = ent_array_map[pos_matrix[:,2]]
            h_1_batch = ent_array_map[neg_matrix[:,0]]
            t_1_batch = ent_array_map[neg_matrix[:,2]]
	   
	    #create a dict mapping from triple score (positive and negative) matrix to class set {0,1} 
	    score_dict = {}
	    true_count = 0
	    #print len(pos_matrix) 	
	    for i in range(len(pos_matrix)): 
		#if abs(f1-f2)<1e-6
		pos_score = score_func(h_batch[i], l_batch[i], t_batch[i])
		#print pos_score
		neg_score = score_func(h_1_batch[i], l_batch[i], t_1_batch[i])

		if pos_score not in score_dict and neg_score not in score_dict and abs(pos_score-neg_score)>=1e-20:
			true_count  +=1
			score_dict[pos_score] = 1
			score_dict[neg_score] = 0
   
            #score_dict = SortedDict(score_dict)
	    #print score_dict.items()   #will print the dictionary ({key: value, ..}
	    score = [score_dict.items()[i][0] for i in range(len(score_dict.items()))]
	    y = [score_dict.items()[i][1] for i in range(len(score_dict.items()))]
            #print score
            #print y
	    return score, y, len(score_dict)


def get_batches(ent_array_map, rel_array_map, pos_matrix, neg_matrix):    
    #randomly corrupt triples from positive matrix resulting in corrupted negative matrix; get the embeddings for s,r,o
    h_batch = ent_array_map[pos_matrix[:,0]]
    l_batch = rel_array_map[pos_matrix[:,1]]
    t_batch = ent_array_map[pos_matrix[:,2]]

    h_1_batch = ent_array_map[neg_matrix[:,0]]
    t_1_batch = ent_array_map[neg_matrix[:,2]]
    return h_batch, l_batch, t_batch, h_1_batch, t_1_batch

# method for creating a map of relation-specific thresholds used in classification 
def find_thresholds(triples_matrix, ent_array_map, rel_array_map):
    thresholds = {}
    start = timeit.default_timer()	
    for i in range(len(triples_matrix)):
        tail = ent_array_map[triples_matrix[i,2]]    
        head = ent_array_map[triples_matrix[i,0]]    
        label = rel_array_map[triples_matrix[i,1]]
	score = score_func(head, label, tail) 
 	if triples_matrix[i,1] in thresholds and thresholds[triples_matrix[i,1]] > score:
	    	thresholds[triples_matrix[i,1]] = score 
        if triples_matrix[i,1] not in thresholds:
	#else:   # this line gives better overall results but misses a recall of 1 somehow...
            	thresholds[triples_matrix[i,1]] = score
    return thresholds


def roc_in_time(KBC_Model, PATH, x, rel_list, triples_matrix, pos_matrix, neg_matrix, num):
    print 'Roc over time at different learning stages for the same relation: '
    for i in range(num):
        ####,
        # load and unpack trained and initial model 

        MODEL_PATH = PATH + model_name + '_model_' + str(i)
        #print MODEL_PATH
        ent_array_map, rel_array_map = KBC_Model.load_model(MODEL_PATH)

        h_batch, l_batch, t_batch, h_1_batch, t_1_batch = get_batches(ent_array_map, rel_array_map, pos_matrix, neg_matrix)

        score, y, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map)
        counts = triples_matrix[:,1].tolist().count(x)
        rel_counts = round(float(counts)/len(triples_matrix), 4)*100

        # ROC-analysis will draw two plots (based on learned and initial embedding) where each each point indicates a different model distinguished by different classification thresholds 
        #roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x], score_init=score_init, y_init=y_init, reverse=False)
        roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x])
        #roc_analysis(score, y, counts, rel_counts, sample_size, title=' ', score_init=None, y_init=None, reverse=False)
        #fpr, tpr, thresholds = metrics.roc_curve(y, score) #, pos_label=1)
        #print metrics.auc(fpr, tpr)

    
def main(arg=None):

    KBC_Model = KBC_Util_Class(dataset, swap, model_name, dim)
    PATH, MODEL_PATH, INITIAL_MODEL = KBC_Model.get_PATHS()

    test_size = 1000
    #test_size = None
    
    triples_matrix = KBC_Model.get_triple_matrix()
    triples_set = set((tuple(triples_matrix[i])) for i in range(len(triples_matrix)))

    '''
    #load set of all triples, train, valid and test data
    triples, train, valid, test = KBC_Model.load_data()
    ent_URI_to_int, rel_URI_to_int = KBC_Model.create_dicts(triples)   #load dicts 
    #load triples_set for faster existential checks, and int_matrices 
    triples_set, train_matrix, valid_matrix, test_matrix  = KBC_Model.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)
    
    #consider the whole triple set for drawing triples to evaluate with classification task
    triples_matrix = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
    '''
   
    # load and unpack trained and initial model 

    ent_array_map, rel_array_map = KBC_Model.load_model(MODEL_PATH) 
    ent_init_array_map, rel_init_array_map = KBC_Model.load_model(INITIAL_MODEL) 

    ent_list, rel_list = KBC_Model.get_int_to_URI()
    n = len(ent_list)
    m = len(rel_list)

    if test_size != None: 
	#randomly sample of test_size from triple set as positive matrix (true triples) 
	selected_indices = np.random.randint(len(triples_matrix), size=test_size)
	pos_matrix = triples_matrix[selected_indices] 
    else:
	pos_matrix = np.copy(triples_matrix)
	 
    neg_matrix = KBC_Model.corrupt_triple_matrix(triples_set, pos_matrix, n)

    h_batch, l_batch, t_batch, h_1_batch, t_1_batch = get_batches(ent_array_map, rel_array_map, pos_matrix, neg_matrix)

    h_init_batch, l_init_batch, t_init_batch, h_1_init_batch, t_1_init_batch = get_batches(ent_init_array_map, rel_init_array_map, pos_matrix, neg_matrix)
   
    #confusion matrix with regard to a single threshold based on the sample matrix of test_size

    thresholds = find_thresholds(triples_matrix, ent_array_map, rel_array_map)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(pos_matrix)):
        if score_func(h_batch[i], l_batch[i], t_batch[i]) < thresholds[pos_matrix[i,1]]:
            fn +=1
        else: 
            tp +=1
            
        if score_func(h_1_batch[i], l_batch[i], t_1_batch[i]) < thresholds[pos_matrix[i,1]]:
            tn +=1
        else: 
            fp +=1
    

    #Accuracy with regard to positive and negative classifications
    acc =  (tp + tn)/float((2*len(pos_matrix)))
    #precision with regard to positive classifications (ratio of truly positive triples from all positively classfied triples)
    prec = tp/float((tp + fp))
    #recall with regard to positive classications (ration of truly positive triples from all actual positive triples) 
    rec = tp/float((tp + fn))
    
    #F-measure as the weighted harmonic mean that assesses the utility tradeoff between Precision and Recall 
    alpha = 0.5  #0.5 if recall and precision are balanced, else will be in favour of either of them 
    F = 1/((alpha/prec)+(1-alpha)/rec)

    print "\nClassification Task\n"
    print "set size of triples to be classified (based on single threshold): {}".format(2*len(pos_matrix))
    print "fn: {}, tp: {}, tn: {}, fp: {}".format(fn, tp, tn, fp)
    print "sum: {}  {}".format(fn+tp+tn+fp, len(pos_matrix)*2) 
    print "Accuracy: {}".format(acc)
    print "Precision: {}".format(prec)
    print "Recall (TP-Rate): {}".format(rec)

    tp_rate = tp/(tp + fn)
    fp_rate = fp/float((fp + tn))
    print "FP-Rate: {}".format(fp_rate)

    # Test roc-analysis, that is, evaluation of different thresholds based on a set of pos and neg triples with common relations:

    #choose a random relation from the test sample to draw 
    x = triples_matrix[np.random.randint(0,len(triples_matrix)),1]    
    max_len = 0
    for r in range(len(rel_list)): 
	cur_len = len(triples_matrix[np.where(triples_matrix[:,1] == r)])
	if  cur_len > max_len: 
		max_len = cur_len
		x = r
	
    print x, len(triples_matrix[np.where(triples_matrix[:,1] == x)])

    pos_matrix = triples_matrix[np.where(triples_matrix[:,1] == x)]


    #if sample size of positive triples larger than a threshold, curtail 
    if len(pos_matrix) > test_size:
	np.random.shuffle(pos_matrix)
        pos_matrix = pos_matrix[0:test_size]


    print "\nROC Analyis for a specific relation with regard to several classification thresholds\nRelation: ", rel_list[x] 
    print "size of classification set with common relations: {}".format(2*len(pos_matrix))
   
    #based on positive set create negative set, either by total random corruption of head or tail or by corrupting with  random entity from domain and range of relation (harder case)    
    #neg_matrix = KBC_Model.corrupt_triple_matrix(triples_set, pos_matrix, n) 
    neg_matrix = KBC_Model.corrupt_triple_matrix_common_rel(triples_set, triples_matrix, pos_matrix, x)

    score, y, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map)
    score_init, y_init, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_init_array_map, rel_init_array_map)
    counts = triples_matrix[:,1].tolist().count(x)
    rel_counts = round(float(counts)/len(triples_matrix), 4)*100
    

    # ROC-analysis will draw two plots (based on learned and initial embedding) where each each point indicates a different model distinguished by different classification thresholds 
    roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x], score_init=score_init, y_init=y_init, reverse=False)

    num = 20
    roc_in_time(KBC_Model, PATH, x, rel_list, triples_matrix, pos_matrix, neg_matrix, num)
 
    # Now conduct a ROC-analysis for every relation based on a randomly drawn set of a max_bound for relation: 
    # Statistics of classfication performance across all relations is reported as a histogram with additional metrics
    auc_list = []
    counts_list = []
    start = timeit.default_timer()
    # iterate over all relations in the triple store 
    for x in range(m): 

            #draw a random sample of triples with current relation upto a theshold size:
	    pos_matrix = triples_matrix[np.where(triples_matrix[:,1] == x)]

	    if len(pos_matrix) > test_size:
		np.random.shuffle(pos_matrix)
		pos_matrix = pos_matrix[0:test_size]


	    #print "\nROC Analyis for a specific relation with regard to several classificaiton thresholds\nRelation: ", rel_list[x] 
	    #print "size of classification set with common relations: {}".format(2*len(pos_matrix))
	    
            #based on positive set create negative set, either by total random corruption of head or tail or by corrupting with  random entity from domain and range of relation (harder case)    
            #neg_matrix = KBC_Model.corrupt_triple_matrix(triples_set, pos_matrix, n) 
	    neg_matrix = KBC_Model.corrupt_triple_matrix_common_rel(triples_set, triples_matrix, pos_matrix, x)
          
	    score, y, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map)
	    score_init, y_init, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_init_array_map, rel_init_array_map)

            # get the relative proportion of the relation in the entire triple set: 
	    counts = triples_matrix[:,1].tolist().count(x)
	    rel_counts = round(float(counts)/len(triples_matrix), 4)*100
	    
            #to show graph 
	    roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x], score_init=score_init, y_init=y_init, reverse=False)

	    #score = [1/score[i] for i in range(len(score))]
	    #score = [-score[i] for i in range(len(score))]
	    #y.reverse()

	    #roc.roc_analysis(score, y, title=' ', reverse=False)
	   
	    
	    #y = np.array([1, 1, 2, 2])

	    # calculate auc (area under the roc-curve) 
	    #fpr, tpr, thresholds = metrics.roc_curve(y, score) #, pos_label=1)
	    #auc_list.append(metrics.auc(fpr, tpr))
            counts_list.append(counts)
            '''
            print rel_list[x]
            print x, counts, auc_list[x], '\n'

            print np.mean(auc_list)
            print np.min(auc_list) 
            print np.max(auc_list) 
            '''
    stop = timeit.default_timer()
    print "time taken roc-analyses on all relations in triple store: {} min".format((stop - start)/60)
        
    #print auc_list 
    count_max = np.max(counts_list)
    histo = np.histogram(auc_list, bins='auto')	
    #print histo 
    # get the maximum frequency of a value to scale the
    max_freq = np.max(histo[0])
    #an attempt to plot the counts of the relations but to see any correlation between learning performance and frequency of relation
    #new_list = np.array(counts_list)/(count_max/max_freq)
    #print new_list
    #plt.plot(np.arange(10), new_list) 	

    # Now plot histogram of all auc-values for every relation and include an info-box with more metrics	
    plt.hist(auc_list, bins='auto')         #bins are the container ranges in which a value uniquely falls
    plt.title("Histogram of AUC-values for Relation-Specific Classification over all Relations")
    first_line = 'number of relations:  '+str(m) + '\n'
    second_line = 'mean auc:  '+str(round(np.mean(auc_list), 3)) + '\n'
    third_line = 'min auc:  '+str(round(np.min(auc_list), 3)) + ' (#: ' + str(auc_list.count(np.min(auc_list)))+  ')\n'
    fourth_line = 'max auc:  '+str(round(np.max(auc_list), 3)) + ' (#: ' + str(auc_list.count(np.max(auc_list)))+  ')' 
    plt.text(0.1, max_freq/float(2), first_line + second_line + third_line + fourth_line, style='italic', bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})
    plt.xlim([0,1])
    plt.show()



if __name__=="__main__": 
    #tf.app.run()
    main()    


