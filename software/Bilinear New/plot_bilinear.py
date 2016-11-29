'''
Script plots the validation results for the trained model in this directory from 'models' folder.
Run the code with following args: 

   python plot_bilinear.py [not normalized]
   python plot_bilinear.py diagonal [not normalized]

'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys

os.chdir(os.getcwd())
DATA_PATH = "../../data/"

model_name = 'bilinear'
MODELS_PATH = 'models/'

#model_name ='bilinear'


def main(arg=None): 
 
    model_name = None

    if len(sys.argv) == 1:
	model_name = 'bilinear'
    if len(sys.argv) == 2:
	if sys.argv[1] == 'diagonal': 
		model_name = 'diagonal'
    if len(sys.argv) == 3:
	if sys.argv[1] == 'not' and sys.argv[2] == 'normalized': 
		model_name = 'bilinear not normalized'
    if len(sys.argv) == 4:
	if sys.argv[1] == 'diagonal' and sys.argv[2] == 'not' and sys.argv[3] == 'normalized':
		model_name = 'diagonal not normalized' 
    if model_name == None:  
    	print "\nInvalid arguments. Run code with valid args:\n\t'python plot_bilinear.py [not normalized]'\n\t'python plot_bilinear.py diagonal [not normalized]'\n"
    	return

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

    print "\nTo plot other trained models, run the code again with following args:\n\t'python plot_bilinear.py [not normalized]'\n\t'python plot_bilinear.py diagonal [not normalized]'\n"

if __name__=="__main__": 
    main()

