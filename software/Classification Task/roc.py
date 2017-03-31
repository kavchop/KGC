import matplotlib.pyplot as plt
import numpy as np



def calc_roc(score, y, num_thr):
		#score = [i for i in reversed(score)]
		#y = [i for i in reversed(y)] 
		roc_x = []
	    	roc_y = []
	    	roc_x_old = []
	    	roc_y_old = []
		rec = []
	    	prec = []
		min_score = min(score)
		max_score = max(score)
		thr = np.linspace(min_score, max_score, num_thr)   #30 thresholds considered!
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
			if TP + FP != 0 and TP + FN !=0:
				prec.append(TP/float(TP+FP))
				rec.append(TP/float(TP + FN)) 
			FP=0
			TP=0
			TN, FN= 0, 0
		return roc_y, roc_x, rec, prec   



def plot_roc_curve(tpr, fpr, counts, rel_counts, sample_size, title = ' ', tpr1=None, fpr1=None):
		plt.plot(tpr, fpr, color='b', linewidth=1.5, label='learned embedding')
		if tpr1!=None: 
			plt.plot(tpr1, fpr1, color='firebrick', linewidth=1.5,  label='initial embedding')
                plt.ylim([-0.025,1.025])
	    	plt.xlim([-0.025,1.025])
		plt.ylabel('TP-Rate')
		plt.xlabel('FP-Rate')
		plt.title('ROC Analysis: \n' +  title, fontsize=15)
		# counts, rel_counts, sample_size, 
		first_line = 'occurances in '+str(counts)+' triples (' + str(rel_counts)+'%)\n'
		second_line = 'sample size of classif. data: ' + str(sample_size)+' (50% pos/neg)'  
		plt.text(0.25, 0.4, 'statistic on this relation:\n\n'+first_line + second_line, style='italic', bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})
		if tpr1!=None:
                	plt.legend(loc='upper right')
		plt.show()
		
def plot_prec_rec_curve(rec, prec, counts, rel_counts, sample_size, title = ' ', rec1=None, prec1=None):
		plt.plot(rec, prec, color='b', linewidth=1.5, label='learned embedding')   #precision recall curve
                if rec1!=None: 
			plt.plot(rec1, prec1, color='firebrick', linewidth=1.5, label='initial embedding')
	    	plt.ylim([-0.025,1.025])
	    	plt.xlim([-0.025,1.025])
                plt.ylabel('Precision')
	        plt.xlabel('Recall')
		plt.title('Precision Recall Curve: \n' +  title, fontsize=15)
                # counts, rel_counts, sample_size, 
		first_line = 'occurances in '+str(counts)+' triples (' + str(rel_counts)+'%)\n'
		second_line = 'sample size of classif. data: ' + str(sample_size)+' (50% pos/neg)'  
		plt.text(0.25, 0.4, 'statistic on this relation:\n\n'+first_line + second_line, style='italic', bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})
		if rec1!=None:
                	plt.legend(loc='upper left')
	    	plt.show() 



def roc_analysis(score, y, counts, rel_counts, sample_size, title=' ', score_init=None, y_init=None, reverse=False):
        num_thr = sample_size * 1
	if reverse:
		score.reverse()
		y.reverse()
		if score != None: 
			score_init.reverse()
			y_init.reverse()
        
	#score = score[::-1]
	#y = y[::-1]

        fpr, tpr, rec, prec = calc_roc(score, y, num_thr)
	if score_init != None: 
		fpr1, tpr1, rec1, prec1 = calc_roc(score_init, y_init, num_thr)
		plot_roc_curve(tpr, fpr, counts, rel_counts, sample_size, title =  title, tpr1=tpr1, fpr1=fpr1)
                plot_prec_rec_curve(rec, prec, counts, rel_counts, sample_size, title = title, rec1=rec1, prec1=prec1)
	else:
		plot_roc_curve(tpr, fpr, counts, rel_counts, sample_size, title =  title)
                plot_prec_rec_curve(rec, prec, counts, rel_counts, sample_size, title =  title)



	'''

	Determine TP, TN, FP, FN  for every threshold and calc for each the tpr = TP/(TP+FN) and fpr = FP/(FP+TN). Plot hem against each other, fpr on the x-axis. Use AUC = area under the curve to get a general performance estimate  (by parametric fit or non-parametric estimate via the trapezoid rule or Mann Whitney U-statistic). There are different ways to get a standard error estimate. We used the SE approximation according to the Hanley-MacNeil paper (1982) but there are better alternatives.
	'''


def main(arg=None):

    score = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0])
    #if lower scores are negative, then don't reverse, else, pass reverse=True
    print score
    print y


    #score = [0.61157854737397666, 0.61333155543438633, 0.61347027907438867, 0.61380663525481627, 0.61387667510504396, 0.61388270070336581, 0.61763564147507721, 0.62229369506021892, 0.62503595900999032, 0.6346244495719946, 0.6411097210479525, 0.6436240434119318, 0.64876986661652747, 0.68092777541602534, 1.575841656231411, 1.5842209731535455, 1.6652834865928168, 1.7982490928434209, 1.9102796187596678, 1.9646356453231126, 2.0365474629343625, 2.0530981325690476, 2.126677143749462, 2.1768506262597138, 2.3955369650415168, 2.5493340348206308]
    #y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #y = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]

    roc_analysis(score, y)
    
if __name__=="__main__": 
    #tf.app.run()
    main()   
