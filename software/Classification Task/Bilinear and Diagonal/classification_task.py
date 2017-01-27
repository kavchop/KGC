import numpy as np
import timeit
import os
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import roc
from sklearn import metrics
from KGC_Model import KGC_Model
import input

model_name = 'bilinear'
dataset = 'Freebase'
dim = 20

os.chdir(os.getcwd())
DATA_PATH = '../../../data/Triple Store/' + dataset + '/'
PATH = '../../../data/Trained Models/'+model_name+'/' + dataset + '/dim = '+str(dim) +'/'

MODEL_PATH = PATH + model_name +'_model'
INITIAL_MODEL = PATH + model_name + '_initial_model'




# This methods corrupts a positive triple set, such that either head or tail is replaced by a random entity from the set of entities
# found in domain or range respectively: e.g. Obama presidentOf Germany as a negative triple  
def corrupt_triple_matrix_common_rel(triples_set, triples_matrix, matrix_batch, x, check_collision=True): 
    corrupt_batch = np.copy(matrix_batch) 

    h_set = triples_matrix[np.where(matrix_batch[:,1] == x)][:,0]
    t_set = triples_matrix[np.where(matrix_batch[:,1] == x)][:,1]
 
    if check_collision: 
        for i in range(len(corrupt_batch)):
            while tuple(corrupt_batch[i]) in triples_set:
                cor_ind = np.random.choice((0,2))  #randomly select index to corrupt (head or tail)
                if cor_ind == 0: #head
                	corrupt_batch[i][cor_ind]= np.random.choice((h_set))
		else: 
			corrupt_batch[i][cor_ind]= np.random.choice((t_set))
    else: 
        for i in range(len(corrupt_batch)):
            a = np.random.randint(n)  #n is number of all unique entities 
            cor_ind = np.random.choice((0,2))
            corrupt_batch[i][cor_ind]= a
    return corrupt_batch


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
		else: 
			true_count  +=1
	    print true_count
   
            #print 'h_batch'
	    #print h_batch
            score_dict = SortedDict(score_dict)
	    #print score_dict.items()   #will print the dictionary ({key: value, ..}
	    score = [score_dict.items()[i][0] for i in range(len(score_dict.items()))]
	    y = [score_dict.items()[i][1] for i in range(len(score_dict.items()))]
	    return score, y, len(score_dict)


def load_model():
    if os.path.isfile(DATA_PATH +'URI_to_int'): 
        URI_to_int = input.pickle_object(DATA_PATH +'URI_to_int', 'r')
        ent_URI_to_int = URI_to_int[0]
        rel_URI_to_int = URI_to_int[1]
        rel_list = rel_URI_to_int.keys() 
	if os.path.isfile(MODEL_PATH):
		model = input.pickle_object(MODEL_PATH, 'r')
		#ent_array_map = model[0]
		#rel_array_map = model[1]
                init_model = input.pickle_object(INITIAL_MODEL, 'r')
		#ent_init_array_map = init_model[0]
		#rel_init_array_map = init_model[1]
                return model, init_model, rel_list
        else: 
                print "Please train a model first before running the cassification task"
		return


# method for creating a map of relation-specific thresholds used in classification 
def find_thresholds(triples_matrix, ent_array_map, rel_array_map):
    thresholds = {}
    start = timeit.default_timer()	
    for i in range(len(triples_matrix)):
        tail = ent_array_map[triples_matrix[i,2]]    
        head = ent_array_map[triples_matrix[i,0]]    
        label = rel_array_map[triples_matrix[i,1]]
	score = score_func(head, label, tail) 
 	if triples_matrix[i,1] in thresholds:
		if thresholds[triples_matrix[i,1]] > score:
	    		thresholds[triples_matrix[i,1]] = score 
        else:
            	thresholds[triples_matrix[i,1]] = score
    return thresholds


#computes the distance score of a triple batch: x=(h+l-t)
def score_func(h, l, t):
    score = np.dot(np.dot(h,l), (np.transpose(t)))
    return int(score)
    
def main(arg=None):
    swap = True
    
    TransE_Model = KGC_Model(dataset, swap, model_name, dim)

    test_size = 3000
    #test_size = None
    
    #load set of all triples, train, valid and test data
    triples, train, valid, test = TransE_Model.load_data()
    ent_URI_to_int, rel_URI_to_int = TransE_Model.create_dicts(triples)   #load dicts 
    #load triples_set for faster existential checks, and int_matrices 
    triples_set, train_matrix, valid_matrix, test_matrix  = TransE_Model.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    #consider the whole triple set for drawing triples to evaluate with classification task
    triples_matrix = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
    
   
    # load and unpack trained and initial model 
    model, init_model, rel_list = load_model() 
    ent_array_map = model[0]
    rel_array_map = model[1]

    ent_init_array_map = init_model[0]
    rel_init_array_map = init_model[1]
    
    n = len(ent_array_map)
    m = len(rel_array_map)

    thresholds = find_thresholds(triples_matrix, ent_array_map, rel_array_map)

    if test_size != None: 
	#randomly sample of test_size from triple set as positive matrix (true triples) 
	selected_indices = np.random.randint(len(triples_matrix), size=test_size)
	pos_matrix = triples_matrix[selected_indices] 
    else:
	pos_matrix = np.copy(triples_matrix)
	 	
    #randomly corrupt triples from positive matrix resulting in corrupted negative matrix; get the embeddings for s,r,o
    h_batch = ent_array_map[pos_matrix[:,0]]
    l_batch = rel_array_map[pos_matrix[:,1]]
    t_batch = ent_array_map[pos_matrix[:,2]]
    print 'label'
    print l_batch.shape       
    neg_matrix = TransE_Model.corrupt_triple_matrix(triples_set, pos_matrix, n)
    h_1_batch = ent_array_map[neg_matrix[:,0]]
    t_1_batch = ent_array_map[neg_matrix[:,2]]
    
    h_init_batch = ent_init_array_map[pos_matrix[:,0]]
    l_init_batch = rel_init_array_map[pos_matrix[:,1]]
    t_init_batch = ent_init_array_map[pos_matrix[:,2]]
            
    h_1_init_batch = ent_init_array_map[neg_matrix[:,0]]
    t_1_init_batch = ent_init_array_map[neg_matrix[:,2]]

    #confusion matrix with regard to a single threshold based on the sample matrix of test_size
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
    '''
    #choose a random relation from the test sample to draw 
    x = triples_matrix[np.random.randint(0,len(triples_matrix)),1]
    pos_matrix = triples_matrix[np.where(triples_matrix[:,1] == x)]

    #if sample size of positive triples larger than a threshold, curtail 
    if len(pos_matrix) > 1000:
	np.random.shuffle(pos_matrix)
        pos_matrix = pos_matrix[0:1000]

    print "\nROC Analyis for a specific relation with regard to several classificaiton thresholds\nRelation: ", rel_list[x] 
    print "size of classification set with common relations: {}".format(2*len(pos_matrix))
   
    # get score list and corresponding target list (of 0s and 1s) and the sample_size considered (will depend on counts of considered relation in the triple store plus ambigious scores due to equality must be eliminated) 
    score, y, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map)
    score_init, y_init, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_init_array_map, rel_init_array_map)

    # get the relative proportion of the relation in the entire triple set: 
    counts = triples_matrix[:,1].tolist().count(x)
    rel_counts = round(float(counts)/len(triples_matrix), 4)*100
    
    # ROC-analysis will draw two plots (based on learned and initial embedding) where each each point indicates a different model distinguished by different classification thresholds 
    roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x], score_init=score_init, y_init=y_init, reverse=False)
 
    neg_matrix = corrupt_triple_matrix_common_rel(triples_set, triples_matrix, pos_matrix, x, check_collision=True)
    score, y, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map)
    score_init, y_init, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_init_array_map, rel_init_array_map)
    counts = triples_matrix[:,1].tolist().count(x)
    rel_counts = round(float(counts)/len(triples_matrix), 4)*100
   

    roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x], score_init=score_init, y_init=y_init, reverse=False)
    '''

    #choose a random relation from the test sample to draw 
    x = triples_matrix[np.random.randint(0,len(triples_matrix)),1]
    pos_matrix = triples_matrix[np.where(triples_matrix[:,1] == x)]

    #if sample size of positive triples larger than a threshold, curtail 
    if len(pos_matrix) > 1000:
	np.random.shuffle(pos_matrix)
        pos_matrix = pos_matrix[0:1000]

    print "\nROC Analyis for a specific relation with regard to several classificaiton thresholds\nRelation: ", rel_list[x] 
    print "size of classification set with common relations: {}".format(2*len(pos_matrix))
   
    #based on positive set create negative set, either by total random corruption of head or tail or by corrupting with  random entity from domain and range of relation (harder case)    
    #neg_matrix = TransE_Model.corrupt_triple_matrix(triples_set, pos_matrix, n) 
    neg_matrix = corrupt_triple_matrix_common_rel(triples_set, triples_matrix, pos_matrix, x, check_collision=True)

    score, y, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_array_map, rel_array_map)
    score_init, y_init, sample_size = get_data_for_roc(pos_matrix, neg_matrix, ent_init_array_map, rel_init_array_map)
    counts = triples_matrix[:,1].tolist().count(x)
    rel_counts = round(float(counts)/len(triples_matrix), 4)*100
    
    # ROC-analysis will draw two plots (based on learned and initial embedding) where each each point indicates a different model distinguished by different classification thresholds 
    roc.roc_analysis(score, y, counts, rel_counts, sample_size, title=rel_list[x], score_init=score_init, y_init=y_init, reverse=False)

    fpr, tpr, thresholds = metrics.roc_curve(y, score) #, pos_label=1)
    print metrics.auc(fpr, tpr)

 
    # Now conduct a ROC-analysis for every relation based on a randomly drawn set of a max_bound for relation: 
    # Statistics of classfication performance across all relations is reported as a histogram with additional metrics
    auc_list = []
    counts_list = []
    start = timeit.default_timer()
    # iterate over all relations in the triple store 
    for x in range(m): 

            #draw a random sample of triples with current relation upto a theshold size:
	    pos_matrix = triples_matrix[np.where(triples_matrix[:,1] == x)]

	    if len(pos_matrix) > 2000:
		np.random.shuffle(pos_matrix)
		pos_matrix = pos_matrix[0:2000]


	    #print "\nROC Analyis for a specific relation with regard to several classificaiton thresholds\nRelation: ", rel_list[x] 
	    #print "size of classification set with common relations: {}".format(2*len(pos_matrix))
	    
            #based on positive set create negative set, either by total random corruption of head or tail or by corrupting with  random entity from domain and range of relation (harder case)    
            #neg_matrix = TransE_Model.corrupt_triple_matrix(triples_set, pos_matrix, n) 
	    neg_matrix = corrupt_triple_matrix_common_rel(triples_set, triples_matrix, pos_matrix, x, check_collision=True)
          
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
	    fpr, tpr, thresholds = metrics.roc_curve(y, score) #, pos_label=1)
	    auc_list.append(metrics.auc(fpr, tpr))
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


