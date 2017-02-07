 

#global- and hyper-parameters

dim = 20                    #dimension of embedding space of entities and relations
shuffle_data = True         #shuffle training data before each epoch
check_collision = True      #when creating corrupt training triples, check if triples is truly non-existent in whole data set 
dataset = 'Freebase'           #Alternatively 'Wordnet', can be extended to any dataset, please make sure to set 'swap' accordingly 
device = 'cpu' 		    # alternatively: gpu
swap = True                 #swap colums of raw triple files to bring to format 'subject predicate object', if not already in this format
max_epoch = 1200             #maximal epoch number for training 
margin = 1                  #param for loss function; greater than 0, e.g. {1, 2, 10}
learning_rate = 0.1         #for Adagrad optimizer
batch_size = 100            #mini-batch-size in every training iteration
result_log_cycle = 5        #run validation and log results after every x epochs
test_size = 1000            #to speed up validation validate on a randomly drawn set of x from validation set
normalize_ent = True  	    #normalization of entity vectors after each gradient step        

