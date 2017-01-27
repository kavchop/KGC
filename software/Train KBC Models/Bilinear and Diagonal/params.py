 
#global- and hyper-parameters

n_red = 20                  #dimension to which original dim n (number of entities) is reduced
shuffle_data = True         #shuffle training data before each epoch
check_collision = True      #when creating corrupt training triples, check if triples is truly non-existent in whole data set 
dataset = 'Freebase'        #alternatively: Wordnet
swap = True                 #swap colums of raw triple files to bring to format 'subject predicate object', if not already in this format
device = 'cpu' 		        # alternatively: gpu
max_epoch = 4000            #maximal epoch number for training 
learning_rate = 0.1        #for Adagrad optimizer
batch_size = 100            #mini-batch-size in every training iteration
result_log_cycle = 250        #run validation and log results after every x epochs
#embedding_log_cycle = 5     #save embedding to disk after every x epochs 
corrupt_two = False         #for each pos triple create two corrupted triples 
test_size = None           #to speed up validation validate on a randomly drawn set of x from validation set
valid_verbose = False       #display scores for each test triple during validation 
train_verbose = False       #display loss after each gradient step during training
normalize_ent = True  	    #normalization of entity matrix after every epoch
