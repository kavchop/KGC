import numpy as np
import tensorflow as tf
from TransE import TransE
import timeit
import os
import sys
import params



def run_training():

    TransE_Model = TransE(params.dataset, params.swap, params.dim, params.margin, params.l1_flag, params.device, params.learning_rate, params.max_epoch, params.batch_size, params.test_size, params.result_log_cycle)

    #myModel = TransE(params.dataset, params.swap, params.dim, params.margin, params.l1_flag, params.device, params.learning_rate, params.max_epoch, params.batch_size, params.test_size, params.result_log_cycle shuffle_data=params.shuffle_data, check_collision=params.check_collision, normalize_ent=params.normalize_ent)

    eval_mode, filtered = TransE_Model.get_eval_tags(sys.argv)

    PATHS, PLOT_PATHS = TransE_Model.getPATHS()
    PATH, MODEL_META_PATH, INITIAL_MODEL, MODEL_PATH, RESULTS_PATH = PATHS[0], PATHS[1], PATHS[2], PATHS[3], PATHS[4]
    PLOT_RESULTS_PATH, PLOT_MODEL_META_PATH = PLOT_PATHS[0], PLOT_PATHS[1]

    # load set of all triples, train, valid and test data
    triples, train, valid, test = TransE_Model.load_data()

    # load dicts (URI strings to int) 
    ent_URI_to_int, rel_URI_to_int = TransE_Model.create_dicts(triples)    

    # load input-formats for script: triples_set (for faster existential checks) and int-matrices for triple stores (train-, test- and valid-set)
    triples_set, train_matrix, valid_matrix, test_matrix  = TransE_Model.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    # entity and relation-lists:
    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations


    # load existing model (that is, model parameters) with given configurations or initialize new and save to disk
    if os.path.isfile(MODEL_PATH):
        print "\n\nExisting TransE model is being loaded...\n"
        ent_array_map, rel_array_map = TransE_Model.load_model(MODEL_PATH)

        # if 'evaluate' tag was passed when running the script, only run evaluation on test-set, save top triples and terminate
	if eval_mode:	 
		evaluate_model(PATH, triples_set, test_matrix, ent_array_map, rel_array_map, filtered)
		return
    else: 
        # case that no trained model with the given configurations exists, but eval_mode=True has been passed 
        if eval_mode:   
		print "\nNo {} model has been trained yet. Please train a model before evaluating.\n".format(model_name)
		return

        # write model configurations and initial model to disk (meta-data on trained model)
        print "\n\nNew TransE model is being initialized and saved before training starts..."
        TransE_Model.save_model_meta(MODEL_META_PATH, PLOT_MODEL_META_PATH)
        ent_array_map, rel_array_map = TransE_Model.init_params(n,m)
        TransE_Model.save_model(INITIAL_MODEL, ent_array_map, rel_array_map)
        

    # open validation-results table to retrieve the last trained epoch
    # if it does not exist, create a new result_table 
    if os.path.isfile(RESULTS_PATH):
        results_table = TransE_Model.pickle_object(RESULTS_PATH, 'r')
        global_epoch = int(results_table[-1][0]) #update epoch_num
        TransE_Model.save_model_meta(MODEL_META_PATH, global_epoch, resumed=True)
    else:
        global_epoch = 0
        results_table, new_record = TransE_Model.update_results_table(RESULTS_PATH, PLOT_RESULTS_PATH, triples_set, valid_matrix, ent_array_map, rel_array_map, global_epoch, 0, init=True)



    # launch TF Session and build computation graph 
    # meta settings passed to the graph 

    g = tf.Graph()
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with g.as_default(), g.device('/'+device+':0'), tf.Session(config=config) as sess:
    '''
    with g.as_default(), g.device('/'+params.device+':0'), tf.Session() as sess: 

        E, R = TransE_Model.get_graph_variables(ent_array_map, rel_array_map)

	h_ph, l_ph, t_ph, h_1_ph, t_1_ph = TransE_Model.get_graph_placeholders()
        h, l, t, h_1, t_1 = TransE_Model.get_model_parameters(E, R, h_ph, l_ph, t_ph, h_1_ph, t_1_ph)

	loss = TransE_Model.get_loss(h, l, t, h_1, t_1)
  
        trainer = TransE_Model.get_trainer(loss)
 
	TransE_Model.model_intro_print(train_matrix)
	
	#op for Variable initialization 
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
  
        #vector X_id mirrors indices of train_matrix to allow inexpensive shuffling before each epoch
        X_id = np.arange(len(train_matrix))

	for i in range(params.max_epoch):
            print "\nepoch: {}".format(global_epoch)
            if params.shuffle_data: 
                np.random.shuffle(X_id)
            start = timeit.default_timer()
            
            loss_sum = 0
            # split the training batch into subbatches; with array_split we will cover all triples even if resulting in uneven batch sizes 
            train_batches = np.array_split(train_matrix[X_id], len(train_matrix)/params.batch_size)
            for j in range(len(train_batches)):

                # get all input batches for current gradient step: 
                # extract h, l and t batches from positive (int) triple batch 
                pos_matrix = train_batches[j]
                h_batch, l_batch, t_batch = pos_matrix[:,0], pos_matrix[:,1], pos_matrix[:,2]
 
                # extract h_1, and t_1 batches from randomly created negative (int) triple batch 
                neg_matrix = TransE_Model.corrupt_triple_matrix(triples_set, pos_matrix, n)
                h_1_batch, t_1_batch = neg_matrix[:,0], neg_matrix[:,2]

                # feed placeholders with current input batches 
                feed_dict={h_ph: h_batch, l_ph: l_batch, t_ph: t_batch, h_1_ph: h_1_batch, t_1_ph: t_1_batch} 
                _, loss_value = sess.run(([trainer, loss]), feed_dict=feed_dict)

                loss_sum += loss_value
            print "total loss of epoch: {}".format(loss_sum)
            # after an epoch decide to normalize entities 
            if params.normalize_ent: 
                sess.run(TransE_Model.normalize_entity_op(E)) 
                '''
                # check if normalization was successful: yes it was :)
                x = E.eval()
                print np.linalg.norm(x, axis=1)
                '''	     
            stop = timeit.default_timer()
            print "time taken for current epoch: {} sec".format((stop - start))
            global_epoch += 1
            if global_epoch > 700:
			params.test_size = len(valid_matrix)-2
			params.result_log_cycle = 25

            #validate model on valid_matrix and save current model after each result_log_cycle
            #if global_epoch == 1 or global_epoch == 10 or global_epoch%result_log_cycle == 0:
            if global_epoch % params.result_log_cycle == 0:
                # extract (numpy) parameters from updated TF variables 
		ent_array_map = E.eval()
                rel_array_map = R.eval()
                results_table, new_record = TransE_Model.update_results_table(RESULTS_PATH, PLOT_RESULTS_PATH, triples_set, valid_matrix, ent_array_map, rel_array_map, global_epoch, loss_sum, results_table)
                # save model to disk only if both h_rank_mean and t_rank_mean improved 
                if min(results_table[1:len(results_table)-1,1]) >= new_record[0,1] and min(results_table[1:len(results_table)-1,2]) >= new_record[0,2]:
		        TransE_Model.save_model(MODEL_PATH, ent_array_map, rel_array_map)
                # print validation results and save results to disk (to two directories where it is accessible for other application, e.g. plotting etc)
                if global_epoch != params.max_epoch:
			print "\n\n******Continue Training******"
	 


def main(arg=None):
    run_training()
    
if __name__=="__main__": 
    #tf.app.run()  # allows a TF-flag-passthrough 
    main()  

