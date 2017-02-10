'''
Author of Code: Kavita Chopra (10.2016)

Description of script: 
- script to visualize TransE embedding in the embedding space before and after training.
- to plot the multi-dimensional embedding of triple elements (head, label, tail) embedding
  dimension needs to be reduced to 2D which is done here through Principle Component Analysis (PCA)
- for embeddings initial_model-embedding created when intializing models before training and 
  trained model are loaded 
- triples to be plotted are triples from top_triples, which is created during evaluation of a 
  successfully trained model and contains triples that are 'true' AND 'highly ranked' by the trained model during evaluation 

Steps for PCA: 
- first normalize data matrix to zero-mean (xi - mean_of_data_X)
- then compute the covariance matrix 
- compute the eigenvalues and eigenvectors of the cov-matrix
- use the eigenvectors correponding to the two largest eigenvectors to project the normalized data to 2D 
  (linear combination between data and eigenvectors) 

'''


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle 
import timeit 
import os
from KBC_Util_Class import KBC_Util_Class



dim = 10
model_name = 'TransE'
dataset = 'Freebase'
swap = True



# Dimensionality Reduction using Principle Component Analysis (PCA)

def PCA(data_matrix, n=2):  #data matrix of column vectors (e.g. 500 x 150, 150 vectors of dim=500)
    # first normalize the data in X to zero mean 
    x_mean = np.mean(data_matrix)
    X = np.subtract(data_matrix,x_mean)
    # compute corresponding covariance of data matrix 
    C = np.cov(X)
    # compute the eigen-decomposition of the cov-matrix
    # eigenvalues, eigenvectors are output in ascending order
    E,U = la.eigh(C)
    # use the n eigenvectors u_1...u_n of the larges eigenvalues to 
    # project (dot-product) the normalized data (X) into R^n 
    P = np.asarray([U[:,U.shape[1]-i] for i in range(1, n+1)])
    return P.dot(X)
    


#plot all n triple embeddings reduced to 2D or only the first num_plot-number of triples from n triples 

def plotData_scatter(n, x, y, title_prefix, num_plot=None, sub_sample=None, rel_sample=None, obj_sample=None):

    if num_plot==None: 
        num_plot = n 

    h_x = x[0:n]
    l_x = x[n:2*n]
    t_x = x[2*n:3*n]

    h_y = y[0:n]
    l_y = y[n:2*n]
    t_y = y[2*n:3*n]
    dist = 0  #distance factor for printing labels for relation or entitiy next to data points in plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([-1,1])
    plt.ylim([-1,1])

    ax.set_title(title_prefix +" TransE Embedding: 2D Scatter View of Entities")
    s = 100

    # first plot a scatter plot to visualize the structure of heads and tails 
    ax.scatter(h_x+t_x, h_y+t_y, c='black',s=s*0.8, alpha=.5, label="all data")   #print all data points in black 

    # print triple elements in different colors:
    ax.scatter(h_x[0:num_plot],h_y[0:num_plot], c='b',s=s, marker='^', alpha=.7, label="sample head")
    ax.scatter(t_x[0:num_plot],t_y[0:num_plot], c='g',s=s, alpha=.7, label="sample tail")

    # annotate the entities 
    a = 0
    if sub_sample != None:
       for i,j in zip(h_x[0:num_plot],h_y[0:num_plot]):
            ax.annotate(sub_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
    a = 0
    if obj_sample != None:
       for i,j in zip(t_x[0:num_plot],t_y[0:num_plot]):
            ax.annotate(obj_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
    plt.legend(loc='upper right')
    plt.show()


#plot all n triple embeddings reduced to 2D or only the first num_plot-number of triples from n triples 

def plotData_rel_type(n, x, y, title_prefix, num_plot=None, sub_sample=None, rel_sample=None, obj_sample=None):

    if num_plot==None: 
        num_plot = n 

    h_x = x[0:n]
    l_x = x[n:2*n]
    t_x = x[2*n:3*n]

    h_y = y[0:n]
    l_y = y[n:2*n]
    t_y = y[2*n:3*n]
    dist = 0  #distance factor for printing labels for relation or entitiy next to data points in plot

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    if rel_sample != None: 
    	ax.set_title(title_prefix +" TransE Embedding - Relation between Entities: \n" + rel_sample[0])

    for i in range(num_plot):
    	#plt.plot([h_x[i],l_x[i]],[h_y[i], l_y[i]], color='firebrick') 
    	plt.plot([h_x[i],t_x[i]],[h_y[i], t_y[i]], color='g') 

        s = 100
        ax.scatter(h_x[i],h_y[i], c='b',s=s, marker='^', alpha=.7)
        #ax.scatter(l_x[i],l_y[i], c='r',s=s, marker='s', alpha=.7)
        ax.scatter(t_x[i],t_y[i], c='r',s=s, alpha=.7)
    
        # to set the label just ones (instead num_plot times): 
        if i == 0:
        	ax.scatter(h_x[i],h_y[i], c='b',s=s, marker='^', alpha=.7, label="head")
                #ax.scatter(l_x[i],l_y[i], c='r',s=s, marker='s', alpha=.7, label="head + label = translated tail")
                ax.scatter(t_x[i],t_y[i], c='r',s=s, alpha=.7, label="tail")

    # annotate the points 
    a = 0
    if sub_sample != None:
       for i,j in zip(h_x[0:num_plot],h_y[0:num_plot]):
            #for i,j in zip(h_x,h_y):
            ax.annotate(sub_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
            ax.scatter(h_x[0:num_plot],h_y[0:num_plot], c='b',s=30) #, label="head")
    '''
    a = 0
    if rel_sample != None:
       for i,j in zip(l_x[0:num_plot],l_y[0:num_plot]):
            ax.annotate(rel_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
            ax.scatter(l_x[0:num_plot],l_y[0:num_plot], c='r',s=30) #, label="label")
    '''
    a = 0
    if obj_sample != None:
       for i,j in zip(t_x[0:num_plot],t_y[0:num_plot]):
            ax.annotate(obj_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
    plt.legend(loc='upper right')
    plt.show()




def plotData_scatter_all(n, x, y, title_prefix, num_plot=None, sub_sample=None, rel_sample=None, obj_sample=None):

    if num_plot==None: 
        num_plot = n 

    h_x = x[0:n]
    l_x = x[n:2*n]
    t_x = x[2*n:3*n]

    h_y = y[0:n]
    l_y = y[n:2*n]
    t_y = y[2*n:3*n]
    dist = 0  #distance factor for printing labels for relation or entitiy next to data points in plot

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    ax.set_title(title_prefix +" TransE Embedding:\n2D Scatter View of Relations")
    s = 100
 
    #ax.scatter(h_x[0:num_plot],h_y[0:num_plot], c='b',s=s, marker='^', alpha=.7, label="head")
    ax.scatter(l_x[0:num_plot],l_y[0:num_plot], c='g',s=s, marker='s', alpha=.7, label="label")
    #ax.scatter(t_x[0:num_plot],t_y[0:num_plot], c='r',s=s, alpha=.7, label="label")


    # annotate the points 
    a = 0
    if sub_sample != None:
       for i,j in zip(h_x[0:num_plot],h_y[0:num_plot]):
            #for i,j in zip(h_x,h_y):
            #ax.annotate(sub_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
    
    a = 0
    if rel_sample != None:
       for i,j in zip(l_x[0:num_plot],l_y[0:num_plot]):
            ax.annotate(rel_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
            a +=1
            #ax.scatter(l_x[0:num_plot],l_y[0:num_plot], c='r',s=s, marker='s', alpha=.7, label="label")
    
    a = 0
    if obj_sample != None:
       for i,j in zip(t_x[0:num_plot],t_y[0:num_plot]):
            #ax.annotate(obj_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
	    #ax.scatter(t_x[i],t_y[i], c='g',s=s, alpha=.7, label="tail")
            a +=1
    plt.legend(loc='upper right')
    plt.show()




def prepare_data_for_PCA(entity_embed, relation_embed, sample, normalize):

    h =  entity_embed[sample[:,0]]
    #l = entity_embed[sample[:,0]] + relation_embed[sample[:,1]]
    l = relation_embed[sample[:,1]]
    t =  entity_embed[sample[:,2]]
    
    if normalize:
        for i in range(len(h)):
            h[i] = h[i]/ np.linalg.norm(h[i])

        for i in range(len(t)):
            t[i] = t[i]/ np.linalg.norm(t[i])

        for i in range(len(l)):
            l[i] = l[i]/ np.linalg.norm(l[i])
	'''
	h = h / np.linalg.norm(h, axis=1)
	l = l / np.linalg.norm(l, axis=1)
	t = t / np.linalg.norm(t, axis=1)

	'''
    
    data = np.concatenate((h, l, t), axis=0)
    data = np.transpose(data) 
    
    return data


# method draws a random sample of triples and returns data in the required format for PCA
def create_data(KBC_Model, n, normalize, rel_type): 

    PATH, MODEL_PATH, INITIAL_MODEL = KBC_Model.get_PATHS()
    triples = KBC_Model.get_triple_matrix()

    sample = get_sample(triples, n, rel_type)

    # Now load initial and final embedding and call prepare_data_for_PCA() method to get the embeddings of the selected triples 

    init_entity_embed, init_relation_embed = KBC_Model.load_model(INITIAL_MODEL)
    
    initial_data = prepare_data_for_PCA(init_entity_embed, init_relation_embed, sample, normalize)
    
    entity_embed, relation_embed = KBC_Model.load_model(MODEL_PATH)
    
    learned_data = prepare_data_for_PCA(entity_embed, relation_embed, sample, normalize)
    
    # return prepared data and lists of h,l,t based on random sample
    return initial_data, learned_data, sample[:,0], sample[:,1], sample[:,2]


def get_sample(triples, n, rel_type): 

    #rel_type = 'many to one'
    #rel_type = 'one to many'
    #rel_type = 1
    #rel_type = 'random_many'
    sample = triples 

    if rel_type == 'one to many':
    	# for one to many
    	length = 0
    	while length < n:
	    # choose random triple
	    x = int(triples[np.random.randint(0,len(triples)),1])

	    # extract all triples with label of random triple
	    h = triples[x,0]
	    l = triples[x,1]

	    sample = triples[np.where(triples[:,0] == h)] 
	    sample = sample[np.where(sample[:,1] == l)] 
	    length = len(sample)


    if rel_type == 'many to one':
    	# for many to one 
    	length = 0
    	while length < n:
	    # choose random triple
	    x = int(triples[np.random.randint(0,len(triples)),1])

	    # extract all triples with label of random triple
	    t = triples[x,2]
	    l = triples[x,1]

	    sample = triples[np.where(triples[:,2] == t)] 
	    sample = sample[np.where(sample[:,1] == l)] 
	    length = len(sample)
    
    # many to one with specific relation:
    if type(rel_type) == int:
    	l = rel_type
    	length = 0
    	while length < n:
	    # extract all triples with label l

	    sample = triples[np.where(triples[:,1] == l)] 
	    sample = sample[np.where(sample[:,2] == sample[0,2])] 
	   
	    length = len(sample)
    
    #simple method: draw random sample of size n 
    #draw a random sample from top_triples
    if rel_type == 'random':
    	# for one to many
    	sample = triples
  
    selected_indices = np.random.randint(len(sample), size=n)
    sample = np.array(sample[selected_indices], dtype=np.int32)

    print '\n\nSample to be plotted: \n'
    print sample
    return sample


def main(arg=None):

    # get parameters from command line interface
	 
    print "\n******Visualization of {} embedding before and after training******\n".format(model_name)
    print "Enter the number of points you want to plot.\nA number between 1 and maximally 500. \nIdeally between 10 and 100 since triples will be chosen randomly and visualization should make sense: "
    num= None
    while type(num) != int or num<1 or num>500:
        num_points = raw_input()
        try:
            num = int(num_points)
        except ValueError:
            print "Please enter a number between 1 and maximally 500: "
    
    n = num           #number of (random) triples that undergo dimension reduction through PCA
    
    num = None
    print "Enter the number of points you want to highlight through color and annotation.\nA number between 1 and {}: ".format(n)
    while type(num) != int or num<1 or num>500:
        num_points = raw_input()
        try:
            num = int(num_points)
        except ValueError:
            print "Please enter a number between 1 and {}: ".format(n)

    num_plot = num  #number of points to be highlighted 

    normalize_inp = None
    while normalize_inp not in set(['y', 'Y', 'n', 'N']): 
     	normalize_inp = raw_input("\nPlot the embedding in a normalized space? [y, n]: ")
        if normalize_inp=='y' or normalize_inp=='Y':
            normalize = True
        if normalize_inp=='n' or normalize_inp=='N':
            normalize = False

    rel_type = None 
    rel_type_inp = None
    while rel_type_inp not in set(['0', '1', '2']): 
     	rel_type_inp = raw_input("\nEnter 0 for a 'random sample',\n1 for a sample based on a single 'one to many' - relation,\nor 2 for a sample based on a single 'many to one' - relation: ")
        if rel_type_inp == '0':
            rel_type = 'random'
        if rel_type_inp == '1':
            rel_type = 'one to many'
	if rel_type_inp == '2':
            rel_type = 'many to one'

    print '\nThe Visualizations are now generated ... '
    
    num = num_points  #first x number of triples of n to be plotted

    KBC_Model = KBC_Util_Class(dataset, swap, model_name, dim)
    entity_list, relation_list = KBC_Model.get_int_to_URI()


    initial_data, learned_data, subject_batch, relation_batch, object_batch = create_data(KBC_Model, n, normalize, rel_type)

    # get dimensionality reduced data from PCA-execution 
    initial_red_data = PCA(initial_data)
    learned_red_data = PCA(learned_data)

    subject_sample = [entity_list[subject_batch[i]] for i in range(n)]
    relation_sample = [relation_list[relation_batch[i]] for i in range(n)]
    object_sample = [entity_list[object_batch[i]] for i in range(n)]
 

    #plot intial and learned embedding without annotation

    I_x = initial_red_data[0,:]
    I_y = initial_red_data[1,:]

    L_x = learned_red_data[0,:]
    L_y = learned_red_data[1,:]


    if rel_type == 'random': 
    	plotData_scatter(n, I_x, I_y, 'Initial', num_plot, sub_sample=subject_sample, rel_sample=relation_sample, obj_sample=object_sample)
    	plotData_scatter(n, L_x, L_y, 'Learned', num_plot, sub_sample=subject_sample, rel_sample=relation_sample, obj_sample=object_sample)


	plotData_scatter_all(n, I_x, I_y, 'Initial', num_plot, sub_sample=subject_sample, rel_sample=relation_sample, obj_sample=object_sample)
    	plotData_scatter_all(n, L_x, L_y, 'Learned', num_plot, sub_sample=subject_sample, rel_sample=relation_sample, obj_sample=object_sample)
    
    if rel_type == 'one to many' or rel_type == 'many to one':
    	plotData_rel_type(n, I_x, I_y, 'Initial', num_plot, sub_sample=subject_sample, rel_sample=relation_sample, obj_sample=object_sample)
    	plotData_rel_type(n, L_x, L_y, 'Learned', num_plot, sub_sample=subject_sample, rel_sample=relation_sample, obj_sample=object_sample)

 
    


if __name__=="__main__": 
    main()
