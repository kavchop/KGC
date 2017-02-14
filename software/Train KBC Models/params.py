 


# KBC general model settings: 

dim = 10                    # dimension of embedding space of entities and relations
shuffle_data = True         # shuffle training data before each epoch
check_collision = True      # when creating corrupt training triples, check if triples is truly non-existent in whole data set 
normalize_ent = True  	    # normalization of entity vectors after each epoch 
dataset = 'Freebase'         # triple store 
swap = True                 # swap colums of raw triple files to bring to format 'subject predicate object', if not already in this format
device = 'cpu:0' 		    # alternatively: gpu
memory = 1		    # memory usage in case gpu is used  
max_epoch = 1200            # maximal epoch number for training 
margin = 1                  # margin in loss function which is imposed between pos and neg score; greater than 0, e.g. {1, 2, 10}
learning_rate = 0.1         # for Adagrad optimizer supporting adaptive 
batch_size = 100            # mini-batch-size for a single training iteration (gradient step) within an epoch
result_log_cycle = 3      # cycle in which validation on current embedding is run and results are logged in the results table
test_size = 200             # to speed up validation in early epochs validate on a smaller subset of the validation set
eval_with_np = False        # option to validate/evaluate based on numpy only (else with Tensorflow, which might be faster on gpu) 




# model-specific settings: 

# for TransE 
l1_flag = True    	    #for dissimilarity measure during training       


# for Bilinear Decomposed: 
dim_hidden = 1             #decomposition rank for relation matrices A and B
