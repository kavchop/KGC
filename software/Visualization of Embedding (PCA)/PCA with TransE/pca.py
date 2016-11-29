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




os.chdir(os.getcwd())
DATA_PATH = "../../../data/"
#DATA_PATH = "data/"

INITIAL_MODEL = 'transe_initial_model'
MODEL_PATH = 'transe_model'
model_name = 'TransE'

#Dimensionality Reduction using Principle Component Analysis (PCA)

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

def plotData(n, x,y, title_prefix, num_plot=None, sub_sample=None, rel_sample=None, obj_sample=None):

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
    ax.set_title(title_prefix +" TransE Embedding in 2D through PCA")
    if sub_sample==None and rel_sample==None and obj_sample==None: 
        ax.scatter(x, y, c='black',s=30, label="all data")   #print all data points in black 

        #print triple elements in different colors:
        ax.scatter(h_x[0:num_plot],h_y[0:num_plot], c='b',s=30, label="sample head")
        ax.scatter(l_x[0:num_plot],l_y[0:num_plot], c='r',s=30, label="sample label")
        ax.scatter(t_x[0:num_plot],t_y[0:num_plot], c='g',s=30, label="sample tail")
        plt.legend(loc='upper right')
    else: 
        # annotate the points 
        a = 0
        if sub_sample != None:
            for i,j in zip(h_x[0:num_plot],h_y[0:num_plot]):
            #for i,j in zip(h_x,h_y):
                ax.annotate(sub_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
                a +=1
            ax.scatter(h_x[0:num_plot],h_y[0:num_plot], c='b',s=30, label="head")
        a = 0
        if rel_sample != None:
            for i,j in zip(l_x[0:num_plot],l_y[0:num_plot]):
            #for i,j in zip(l_x,l_y):
                ax.annotate(rel_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
                a +=1
            ax.scatter(l_x[0:num_plot],l_y[0:num_plot], c='r',s=30, label="label")
        a = 0
        if obj_sample != None:
            for i,j in zip(t_x[0:num_plot],t_y[0:num_plot]):
            #for i,j in zip(t_x,t_y):
                ax.annotate(obj_sample[a], (i+a*dist,j+a*dist)) #for u in range(ind)
                a +=1
            ax.scatter(t_x[0:num_plot],t_y[0:num_plot], c='g',s=30, label="tail")

        plt.legend(loc='upper right')

    plt.show()




#takes triples store, returns unique list of entities and relations
def parse_triple_store(triples):    
    e1 = triples[:,0]
    e2 = triples[:,1]
    rel = triples[:,2]
    entity_list = list(set(e1).union(set(e2)))
    relation_list = list(set(rel))
    return entity_list, relation_list

def extract_data(entity_embed, relation_embed, sample, normalize):
 
    #print entity_embed.keys()[0:10]
    file = open(DATA_PATH+'URI_to_int', 'r')
    URI_to_int = pickle.load(file)
    entity_list = URI_to_int[0]
    relation_list = URI_to_int[1]
    file.close()

    # entity and relation lists int_to_URI
    entity_list = entity_list.keys()
    relation_list = relation_list.keys()

    h =  np.asarray([entity_embed[sample[i, 0]] for i in range(len(sample))])
    l =  np.asarray([relation_embed[sample[i,1]] for i in range(len(sample))])
    t =  np.asarray([entity_embed[sample[i,2]] for i in range(len(sample))])


    #h =  np.asarray([entity_embed[entity_list[sample[i, 0]]] for i in range(len(sample))])
    #l =  np.asarray([relation_embed[relation_list[sample[i,1]]] for i in range(len(sample))])
    #t =  np.asarray([entity_embed[entity_list[sample[i,2]]] for i in range(len(sample))])


    if normalize:
        for i in range(len(l)):
            h[i] = h[i]/ np.linalg.norm(h[i], ord=1)

        for i in range(len(l)):
            t[i] = t[i]/ np.linalg.norm(t[i], ord=1)

        for i in range(len(l)):
            l[i] = l[i]/ np.linalg.norm(l[i], ord=1)

    data = np.concatenate((h, l, t), axis=0)
    data = np.transpose(data)
    return data



def create_data(n, normalize=False):
    file = open('top_triples', 'r')
    triples = pickle.load(file)
    file.close()
    print triples

    #draw a random sample of from top_triples (highly ranked true triples)
    selected_indices = np.random.randint(len(triples), size=n)
    sample = np.array(triples[selected_indices], dtype=np.int32)
    # print sample 

    #now load initial and final embedding and call extract_data() method to 
    #create the embeddings of the selected triples 
    file = open(INITIAL_MODEL, 'r')
    transE_model = pickle.load(file)
    file.close()
    init_entity_embed = transE_model[0]
    init_relation_embed = transE_model[1]
    #print init_entity_embed
    
    initial_data = extract_data(init_entity_embed, init_relation_embed, sample, normalize)
    
    file = open(MODEL_PATH, 'r')
    transE_model = pickle.load(file)
    file.close()
    entity_embed = transE_model[0]
    relation_embed = transE_model[1]
    
    learned_data = extract_data(entity_embed, relation_embed, sample, normalize)
    
    return initial_data, learned_data, sample[:,0], sample[:,1], sample[:,2]





def main(arg=None):

    file = open('top_triples', 'r')
    triples = pickle.load(file)
    file.close()

	 
    print "\n******Visualization of {} embedding before and after training******\n".format(model_name)
    print "Enter the number of points you want to plot.\nA number between 1 and maximally {}. \nIdeally between 20 and 50 since triples will be chosen randomly and visualization should make sense: ".format(len(triples))
    num= None
    while type(num) != int or num<1 or num>len(triples):
        num_points = raw_input()
        try:
            num = int(num_points)
        except ValueError:
            print "Please enter a number between 1 and maximally {}: ".format(len(triples))
    
    n = num           #number of (random) triples that undergo dimension reduction through PCA
    
    num= None
    print "Enter the number of points you want to highlight through color and annotation.\nA number between 1 and {}: ".format(n)
    while type(num) != int or num<1 or num>len(triples):
        num_points = raw_input()
        try:
            num = int(num_points)
        except ValueError:
            print "Please enter a number between 1 and {}: ".format(len(triples))

    num_plot = num  #number of points to be highlighted 

    normalize_inp = None
    while normalize_inp not in set(['y', 'Y', 'n', 'N']): 
     	normalize_inp = raw_input("\nPlot the embedding in a normalized space? [y, n]: ")
        if normalize_inp=='y' or normalize_inp=='Y':
            normalize = True
        if normalize_inp=='n' or normalize_inp=='N':
            normalize = False

    num = num_points  #first x number of triples of n to be plotted

    file = open(DATA_PATH+'URI_to_int', 'r')
    URI_to_int = pickle.load(file)
    ent_URI_to_int = URI_to_int[0]
    rel_URI_to_int = URI_to_int[1]
    file.close()
    
    entity_list = ent_URI_to_int.keys()
    relation_list = rel_URI_to_int.keys()

    initial_data, learned_data, subject_batch, relation_batch, object_batch = create_data(n, normalize)
    initial_red_data = PCA(initial_data)
    learned_red_data = PCA(learned_data)

    subject_sample = [entity_list[subject_batch[i]] for i in range(n)]
    relation_sample = [relation_list[relation_batch[i]] for i in range(n)]
    object_sample = [entity_list[object_batch[i]] for i in range(n)]

    #plot intial and learned embedding without annotation
    plotData(n, initial_red_data[0,:],initial_red_data[1,:], 'Initial', num_plot)
    plotData(n, learned_red_data[0,:],learned_red_data[1,:], 'Learned', num_plot)
    
     #plot intial and learned embedding of labels with annotation
     #default hard coding is labels here, but can be changed to or extended by entity annotation (whose URIs however are not in natural language)  
    plotData(n, initial_red_data[0,:],initial_red_data[1,:], 'Initial', num_plot,  rel_sample=relation_sample)
    plotData(n, learned_red_data[0,:],learned_red_data[1,:], 'Learned', num_plot, rel_sample=relation_sample)

if __name__=="__main__": 
    main()
