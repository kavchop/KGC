'''
Script plots the validation results for the trained model in this directory from 'models' folder 

'''

import pickle
import numpy as np
import matplotlib.pyplot as plt



model_name = 'transe'
MODELS_PATH = 'models/'


def main(arg=None): 
	try:
		file_ = open(MODELS_PATH+model_name+'_results','r')
	except IOError: 
 		print "\nNo {} model has been trained yet. Please train a model before plotting.\n".format(model_name)
                return	
	file_ = open(MODELS_PATH+model_name+'_results','r')
	results = pickle.load(file_)    #list of tuples for pos_map and neg_map
	file_.close()

	lr_x = None

	print '\n***** {} model *****\n'.format(model_name)
	print results[0,0:3], '\n'

	results = np.array(results[1:len(results), 0:3], dtype=np.int32)
	print results

	epoch = results[:,0]
	h_mean = results[:,1]
	t_mean = results[:,2]

	plt.ylim([0,max(max(h_mean)+500, max(t_mean)+500)])
	plt.ylabel('mean rank')
	plt.xlabel('epoch')
	if lr_x != None: 
		plt.axvline(x=lr_x, linewidth=1.5, color='k')
		plt.annotate("new learn rate", (lr_x+2,3000))
	plt.plot(epoch, h_mean, color='b', label='rank mean from head substitution')
	plt.plot(epoch, t_mean, color='r', label='rank mean from tail substitution')
	plt.title(model_name+' model - validation results during training')
	plt.legend(loc='upper right')
	first_line = str(h_mean[-2]) + '     ' + str(t_mean[-2])
	second_line = str(h_mean[-1]) + '     ' + str(t_mean[-1])
	plt.text(0.75*max(epoch), 5000, 'h rank  t rank\n...         ...\n'+first_line+'\n'+second_line, style='italic', bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})
	plt.show()

if __name__=="__main__": 
    main()

