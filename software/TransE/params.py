 

#global- and hyper-parameters

dim = 20                    #dimension of embedding space of entities and relations
shuffle_data = True         #shuffle training data before each epoch
check_collision = True      #when creating corrupt training triples, check if triples is truly non-existent in whole data set 
swap = True                #swap colums of raw triple files to bring to format 'subject predicate object', if not already in this format
max_epoch = 1500             #maximal epoch number for training 
global_epoch = 0            #number of epochs model was trained, global across multiple training sessions of the same model
margin = 2                  #param for loss function; greater than 0, e.g. {1, 2, 10}
learning_rate = 0.01         #for Adagrad optimizer
batch_size = 100            #mini-batch-size in every training iteration
l1_flag = True    	    #for dissimilarity measure during training
result_log_cycle = 10        #run validation and log results after every x epochs
embedding_log_cycle = 10     #save embedding to disk after every x epochs
valid_size = 600           #to speed up validation validate on a randomly drawn set of x from validation set
valid_verbose = False       #display scores for each test triple during validation 
train_verbose = False       #display loss after each gradient step during training
normalize_ent = True  	    #normalization of entity vectors after each gradient step        

