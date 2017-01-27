import numpy as np
import timeit
from input import pickle_object
import input
import os
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import eval

os.chdir(os.getcwd())
DATA_PATH = "../../data/"
MODEL_PATH = 'models/bilinear_model'
INITIAL_MODEL_PATH = 'models/bilinear_initial_model'

corrupt_two=False
check_collision=True
    
def main(arg=None):
    swap = True
    l1_flag= True
    
    #test_size = 10000
    test_size = None
    #get the relation-specific thresholds obtained during validation 
    if os.path.isfile('classification thresholds'):
        thresholds = pickle_object('classification thresholds', 'r')
    else: 
    	print "There is no file 'classification thresholds' in your current directory. Please Train a new model and then come back to the classification task."
        return
    
    #load set of all triples, train, valid and test data
    triples, train, valid, test = input.load_data(swap)
    ent_URI_to_int, rel_URI_to_int = input.create_dicts(triples)   #load dicts 
    #load triples_set for faster existential checks, and int_matrices 
    triples_set, train_matrix, valid_matrix, test_matrix  = input.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    #consider the whole triple set for drawing triples to evaluate with classification task
    triples_matrix = np.concatenate((train_matrix, valid_matrix, test_matrix), axis=0)
    
    if os.path.isfile(DATA_PATH +'URI_to_int'): 
        URI_to_int = pickle_object(DATA_PATH +'URI_to_int', 'r')
        ent_URI_to_int = URI_to_int[0]
        rel_URI_to_int = URI_to_int[1]
        rel_list = rel_URI_to_int.keys() 
	if os.path.isfile(MODEL_PATH):
		bilinear_model = input.pickle_object(MODEL_PATH, 'r')
        	W_param = bilinear_model[0]
        	rel_array_map = bilinear_model[1]
		ent_array_map = input.learned_ent_embed(W_param) 
                bilinear_model = pickle_object(INITIAL_MODEL_PATH, 'r')
		W_param = bilinear_model[0]
        	rel_init_array_map = bilinear_model[1]
		ent_init_array_map = input.learned_ent_embed(W_param) 
        else: 
		return 

  	
    n = len(ent_array_map)
    m = len(rel_array_map)

    if test_size != None: 
	#randomly sample of test_size from triple set as positive matrix (true triples) 
	selected_indices = np.random.randint(len(triples_matrix), size=test_size)
	pos_matrix = triples_matrix[selected_indices] 
    else:
	pos_matrix = np.copy(triples_matrix)
	
    '''
    #set of relations for which we have threhsolds obtained during validation 
    known_relations = set(thresholds.keys())

    #filter positive matrix to get triples where relations have thresholds in threshold file
    filtered_matrix = np.reshape(np.array([]), (0,3))
    for i in range(len(pos_matrix)):
        if pos_matrix[i,1] in known_relations: 
	    filtered_matrix = np.append(filtered_matrix, np.reshape(pos_matrix[i], (1,3)), axis=0)
        #else: 
	    #print pos_matrix[i,1]    
    pos_matrix = filtered_matrix[:]
    '''  	
    #randomly corrupt triples from positive matrix resulting in corrupted negative matrix; get the embeddings for s,r,o
    h_batch, l_batch, t_batch = input.create_triple_array_batches(pos_matrix, ent_array_map, rel_array_map)
    #neg_matrix = input.corrupt_triple_matrix(triples_set, pos_matrix, n)
    neg_matrix = input.create_corrupt_matrix(triples_set, corrupt_two, pos_matrix, n, check_collision)
    h_1_batch, t_1_batch = input.create_triple_array_batches(neg_matrix, ent_array_map, rel_array_map, corrupt=True)
    
    h_init_batch, l_init_batch, t_init_batch = input.create_triple_array_batches(pos_matrix, ent_init_array_map, rel_init_array_map)
    h_1_init_batch, t_1_init_batch = input.create_triple_array_batches(neg_matrix, ent_init_array_map, rel_init_array_map, corrupt=True)


    #do a ROC analyis with a single threshold for every instance in test sample

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    
    for i in range(len(pos_matrix)):
        #first evaluate correct triples
        if eval.score_func(h_init_batch[i], l_init_batch[i], t_init_batch[i]) > thresholds[pos_matrix[i,1]]:
            fn +=1
        else: 
            tp +=1
            
        if eval.score_func(h_1_init_batch[i],l_init_batch[i], t_1_init_batch[i]) > thresholds[pos_matrix[i,1]]:
            tn +=1
        else: 
            fp +=1
    
    tp_ = 0
    tn_ = 0
    fp_ = 0
    fn_ = 0
    
    
    for i in range(len(pos_matrix)):
        #first evaluate correct triples
        if eval.score_func(h_init_batch[i], l_init_batch[i], t_init_batch[i]) > thresholds[pos_matrix[i,1]]:
            fn_ +=1
        else: 
            tp_ +=1
            
        if eval.score_func(h_1_init_batch[i],l_init_batch[i], t_1_init_batch[i]) > thresholds[pos_matrix[i,1]]:
            tn_ +=1
        else: 
            fp_ +=1

	
    #print "just a check"
    #print 2*len(pos_matrix), fn+tp+tn+fp
    #Accuracy with regard to positive and negative classifications
    acc =  (tp + tn)/float((2*len(pos_matrix)))
    #precision with regard to positive classifications (ratio of truly positive triples from all positively classfied triples)
    prec = tp/float((tp + fp))
    #recall with regard to positive classications (ration of truly positive triples from all actual positive triples) 
    rec = tp/float((tp + fn))
    
    #F-measure as the weighted harmonic mean that assesses the utility tradeoff between Precision and Recall 
    alpha = 0.5  #0.5 if recall and precision are balanced, else will be in favour of either of them 
    #F = 1/((alpha/prec)+(1-alpha)/rec)
    

    print "\nClassification Task\n"
    print "set size of triples to be classified: {}".format(2*len(pos_matrix))
    print "fn: {}, tp: {}, tn: {}, fp: {}".format(fn, tp, tn, fp)
    print "Accuracy: {}".format(acc)
    print "Precision: {}".format(prec)
    print "Recall (TP-Rate): {}".format(rec)

    tp_rate = tp/(tp + fn)
    fp_rate = fp/float((fp + tn))
    print "FP-Rate: {}".format(fp_rate)
    


    #choose a random relation from the test sample to draw 
    x = triples_matrix[np.random.randint(0,len(triples_matrix)),1]
    pos_matrix = triples_matrix[np.where(triples_matrix[:,1] == x)]
   

    if len(pos_matrix) > 1000:
	np.random.shuffle(pos_matrix)
        pos_matrix = pos_matrix[0:1000]
    print "\nROC Analyis for a specific relation with regard to several classificaiton thresholds\nRelation: ", rel_list[x] 
    print "size of classification set with common relations: {}".format(2*len(pos_matrix))
    
    #get the embedding for s,r,o and get a randomly corrupted negative matrix from the positive matrix
    h_batch, l_batch, t_batch = input.create_triple_array_batches(pos_matrix, ent_array_map, rel_array_map)
    #neg_matrix = input.corrupt_triple_matrix(triples_set, pos_matrix, n)
    neg_matrix = input.create_corrupt_matrix(triples_set, corrupt_two, pos_matrix, n, check_collision)
    h_1_batch, t_1_batch = input.create_triple_array_batches(neg_matrix, ent_array_map, rel_array_map, corrupt=True)
    
    #create a dict mapping from triple score (positive and negative) matrix to class set {0,1} 
    score_dict = {}	
    for i in range(len(pos_matrix)): 
        score_dict[eval.score_func(h_batch[i], l_batch[i], t_batch[i])] = 1
        score_dict[eval.score_func(h_batch[i], l_batch[i], t_batch[i])] = 0
    
    #sort dictionary by keys (scores) to illustrate a possible linear separability 
    score_dict = SortedDict(score_dict)

    #print score_dict.items()   #will print the dictionary ({key: value, ..}
    score = [score_dict.items()[i][0] for i in range(len(score_dict.items()))]
    y = [score_dict.items()[i][1] for i in range(len(score_dict.items()))]
    score = [1/score[i] for i in range(len(score))]
    print "\nsorted scores of triples in the classification with equal number of positive and negative triples:\n"    
    #print score
    print y
    roc_x = []
    roc_y = []
    roc_x_old = []
    roc_y_old = []
    prec = []
    min_score = min(score)
    max_score = max(score)
    thr = np.linspace(min_score, max_score, 30)   #30 thresholds considered!
    FP=0
    TP=0
    FN =0
    TN = 0
    N = sum(y) 
    P = len(y) - N 

    
    for (i, T) in enumerate(thr):
    	for i in range(0, len(score)):   #for every threshold iterate over score vector
		if (score[i] > T):       #starting from low to high shift threshold marker: evaluate all scores against current threshold, everything left of threshold is classified as negative, right positive
		    if (y[i]==1):        #if current score is greater than threshold and if it is classified as pos, then increment tp
		        TP = TP + 1
   
		    if (y[i]==0):        #if it is classified as neg, then increment fp 
		        FP = FP + 1
                else: 
                    if (y[i]==1):        #if current score is greater than threshold and if it is classified as pos, then increment tp
		        FN = FN + 1
                              
		    if (y[i]==0):          #if it is classified as neg, then increment fp 
		        TN = TN + 1
	roc_x_old.append(FP/float(N))      #=fp/tp + fn
	roc_y_old.append(TP/float(P))      #=tp/fp +tn
        roc_x.append(FP/float(FP + TN))     #=fp/tp + fn
	roc_y.append(TP/float(TP + FN))    #=tp/fp +tn (recall)
        if TP + FP ==0:
 	       prec.append(0)
	else: 
        	prec.append(TP/(TP+FP))
	FP=0
	TP=0
        TN, FN= 0, 0
    
    print "\nROC curve with FP-rate as x-axis and TP-rate as y-axis\n"
    print "average FP-rate: {}, max FP-rate: {}".format(np.mean(roc_x), np.max(roc_x))
    print "average TP-rate: {}, max TP-rate: {}\n".format(np.mean(roc_y), np.max(roc_y))

    print "new      old (first two means, last max" 
    print np.mean(roc_x), np.mean(roc_x_old)
    print np.mean(roc_y), np.mean(roc_y_old)
    print np.max(roc_x), np.max(roc_x_old)
    print np.max(roc_y), np.max(roc_y_old)

    print "\nPrecision and recall"
    print roc_y
    print prec
    print roc_y_old
    plt.plot(roc_x, roc_y)
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show()
    plt.plot(roc_x_old, roc_y_old)
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show() 
    '''
    plt.plot(prec, roc_y)   #precision recall curve
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show() 
    plt.plot(prec, roc_y_old)   #precision recall curve
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show() 
    '''
    #control group

    #print score_dict.items()   #will print the dictionary ({key: value, ..}
    score = [score_dict.items()[i][0] for i in range(len(score_dict.items()))]
    #y = [score_dict.items()[i][1] for i in range(len(score_dict.items()))]
    #score = [1/score[i] for i in range(len(score))]
    #print "\nsorted scores of triples in the classification with equal number of positive and negative triples:\n"    
    #print score
    #print y
    roc_x = []
    roc_y = []
    roc_x_old = []
    roc_y_old = []
    prec = []
    min_score = min(score)
    max_score = max(score)
    thr = np.linspace(min_score, max_score, 30)   #30 thresholds considered!
    FP=0
    TP=0
    FN =0
    TN = 0
    N = sum(y) 
    P = len(y) - N 

    
    for (i, T) in enumerate(thr):
    	for i in range(0, len(score)):   #for every threshold iterate over score vector
		if (score[i] < T):       #starting from low to high shift threshold marker: evaluate all scores against current threshold, everything left of threshold is classified as negative, right positive
		    if (y[i]==1):        #if current score is greater than threshold and if it is classified as pos, then increment tp
		        TP = TP + 1
   
		    if (y[i]==0):        #if it is classified as neg, then increment fp 
		        FP = FP + 1
                else: 
                    if (y[i]==1):        #if current score is greater than threshold and if it is classified as pos, then increment tp
		        FN = FN + 1
                              
		    if (y[i]==0):          #if it is classified as neg, then increment fp 
		        TN = TN + 1
	roc_x_old.append(FP/float(N))      #=fp/tp + fn
	roc_y_old.append(TP/float(P))      #=tp/fp +tn
        roc_x.append(FP/float(FP + TN))     #=fp/tp + fn
	roc_y.append(TP/float(TP + FN))    #=tp/fp +tn (recall)
        if TP + FP ==0:
 	       prec.append(0)
	else: 
        	prec.append(TP/(TP+FP))
	FP=0
	TP=0
        TN, FN= 0, 0
    
    print "\nROC curve with FP-rate as x-axis and TP-rate as y-axis\n"
    print "average FP-rate: {}, max FP-rate: {}".format(np.mean(roc_x), np.max(roc_x))
    print "average TP-rate: {}, max TP-rate: {}\n".format(np.mean(roc_y), np.max(roc_y))

    print "new      old (first two means, last max" 
    print np.mean(roc_x), np.mean(roc_x_old)
    print np.mean(roc_y), np.mean(roc_y_old)
    print np.max(roc_x), np.max(roc_x_old)
    print np.max(roc_y), np.max(roc_y_old)

    print "\nPrecision and recall"
    print roc_y
    print prec
    print roc_y_old
    plt.plot(roc_x, roc_y)
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show()
    plt.plot(roc_x_old, roc_y_old)
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show() 
    '''
    plt.plot(prec, roc_y)   #precision recall curve
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show() 
    plt.plot(prec, roc_y_old)   #precision recall curve
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.show() 
    '''      



if __name__=="__main__": 
    #tf.app.run()
    main()    


