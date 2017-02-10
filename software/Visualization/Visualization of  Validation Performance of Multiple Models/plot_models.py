'''
Author: Kavita Chopra (2016, version 1.0)

Script for plotting validation results from Knowledge Graph Completion models.
Based on interactive command line options plots single and multiple models from 'models/' directory

single model: 
- command line options
  - merge head and tail rank mean 
  - plot horizontal line at epoch number x to mark a change in learning rate during
    training   

multiple models: 
- plots up to 4 models from the set ['bilinear', 'diagonal', 'transe', 'bilinear not   normalized', 'diagonal not normalized', 'transe not normalized']  
- normalization refers to employed normalization on entity embedding after every training epoch

'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import os



dim = 10
dataset = 'Freebase'

PLOT_PATH = '../../../data/Model Validation Results for Plotting/' + dataset + '/dim = '+str(dim) +'/'
#PLOT_RESULTS_PATH = PLOT_PATH + model_name + '_results'


def plot_one(models, merge, lr_change=None):
    model = models[0]
    try:
    	file = open(PLOT_PATH + model + '_results','r')
    except IOError:
	print "\nOops: Your input '{}' does not exist in the 'Trained Models' - Directory.\n".format(model)
	return
    results = pickle.load(file)    #list of tuples for pos_map and neg_map
    file.close()
    print "\n*****model: {}*****\n".format(model)
    results = results[:,0:3]
    print results[0,:], '\n'
    results = np.array(results[1:len(results)], dtype=np.int32)  #np.int32
    print results
    epoch = results[:,0]
    h_mean = results[:,1]
    t_mean = results[:,2]
    plt.ylim([0,max(max(h_mean)+500, max(t_mean)+500)])
    #plt.ylim([0,8000])
    plt.ylabel('mean rank')
    plt.xlabel('epoch')
    if merge: 
        title= 'Comparison of Models'
        mean = (h_mean + t_mean)/2
        plt.plot(epoch, mean, color='b', label='rank mean from h/t substitution')
    
    else: 
        title= model + ' model - validation results during training' 
        plt.plot(epoch, h_mean, color='b', label='rank mean from head substitution')
        plt.plot(epoch, t_mean, color='r', label='rank mean from tail substitution')

    if lr_change != None: 
    	plt.axvline(x=lr_change, linewidth=1.5, color='k')
    	plt.annotate("new learn rate", (lr_change+epoch[-1]*0.012,4500))
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()



def plot_many(models):
    color_list = ['b', 'g', 'r', 'k']
    #shuffle color list so that each time you run the code plots will be in different colors
    shuffle(color_list)
    results, results_common = [], []   #results cut to common epoch
    epochs = []
    means = []   #average mean of h_mean and t_mean
    h_means, t_means = [], []
    for i in range(len(models)): 	
        #model = models[i]
        try: 
        	file = open(PLOT_PATH + models[i] + '_results','r')
        except IOError:
		print "\nOops: Your input '{}' does not exist in the 'Trained Models' Directory.\n".format(models[i])
		return
	 	
        results.append(pickle.load(file))    #list of tuples for pos_map and neg_map
        file.close()
        #print np.asarray(results, dtype=np.object) 
        
        header = results[i][0,0:3]
        results[i] = np.array(results[i][1:len(results[i]), 0:3], dtype=np.int32)
	
    min_length = min([len(results[i]) for i in range(len(models))])
    epoch = results[0][0:min_length,0]	    
    
    #first plot results curtailed to the common minimum length
    for i in range(len(models)):
        results_common.append(results[i][0:min_length,:])
        print "\n*****model: {}*****\n".format(models[i])
        print header, '\n'
        print results[i]
        h_means.append(results_common[i][:,1])
        t_means.append(results_common[i][:,2])
        means.append((h_means[i] + t_means[i])/2)
        plt.plot(epoch, means[i], color=color_list[i], label=models[i]) 
    
    title = 'Comparison of Models - common epochs'
    plt.ylim([0,8000])
    plt.ylabel('mean rank')
    plt.xlabel('epoch')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()
    
    #then plot results from all epochs
   
    h_means, t_means = [], []
    means = []

    min_length = min([len(results[i]) for i in range(len(models))])
    epoch = results[0][0:min_length,0]
    for i in range(len(models)):
        #results[i] = results[i][0:min_length,:]
	epochs.append(results[i][:,0])
        h_means.append(results[i][:,1])
        t_means.append(results[i][:,2])
        means.append((h_means[i] + t_means[i])/2)
        plt.plot(epochs[i], means[i], color=color_list[i], label=models[i]) 
    
    title = 'Comparison of Models - all epochs'
    plt.ylim([0,8000])
    plt.ylabel('mean rank')
    plt.xlabel('epoch')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()



    
    
def main(arg=None):
    
    merge = False 
    lr_change = None
    model_list = ['Bilinear', 'Diagonal', 'TransE', 'Bilinear Decomp']
    #normed_list = [model_list[i] + ' not normalized' for i in range(len(model_list))]
    #model_set = (set(model_list)).union(set(normed_list))
    model_set = set(model_list) 
    

    print "\n*****Plotting Knowledge Graph Completion Models*****\n"
    print "This is the set of models you can plot: \n"
    print model_list, '\n' 
    print "Make sure the models you enter are in the 'models'-folder in the same directory.\nHere is a view of the content of the 'models'-folder: \n"
    print os.listdir(PLOT_PATH), '\n'
    num= None
    while type(num) != int or num<1 or num>4:
        num_models = raw_input("Enter the number of models you want to plot [1 - 4]: ")
        try:
            num = int(num_models)
        except ValueError:
            print "Please enter a number between 1 and 4\n"

    models = []
    num_models = num

    print "Now enter the model(s): \n"
    while len(models) < num_models:
        cur_model = raw_input("model {}: ".format(len(models)+1))
	cur_model = cur_model.strip()
        if cur_model in model_set: 
            models.append(cur_model)
            models = list(set(models))

    if num_models > 1:
        print "\nThese correctly entered models will be plotted: "
        print models, '\n'


    if num_models==1:
        merge_inp = None
	lr_inp = None
	num = None
        while merge_inp not in set(['y', 'Y', 'n', 'N']): 
            merge_inp = raw_input("\nDo you want to merge the head and tail mean ranks? [y, n]: ")
            if merge_inp=='y' or merge_inp=='Y':
                merge = True
            if merge_inp=='n' or merge_inp=='N':
                merge = False
	while lr_inp not in set(['y', 'Y', 'n', 'N']): 
            lr_inp = raw_input("\nDid the learning rate change during training? [y, n]: ")
	    if lr_inp=='n' or lr_inp=='N':
                lr_change = None   
            if lr_inp=='y' or lr_inp=='Y':
    	    	while type(num) != int or num<1:
			num = raw_input("\nAt what epoch did the learning rate change?: ")
			try:
		    		num = int(num)
			except ValueError:
		    		print "Please enter an integer number > 1\n"
	    	lr_change = num

          
	plot_one(models, merge, lr_change)

    else: 
        plot_many(models)
    
if __name__=="__main__": 
    main()
