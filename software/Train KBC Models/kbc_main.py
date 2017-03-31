import sys
import params




def process_sys_argv():

	bi_diagonal = False
	if len(sys.argv) < 2: 
		print "\n\n   Type python kbc_main.py ...\n"
		print '   ... transe\n   ... bilinear [diagonal, decomposed]\n\n'
		return 

	
	if sys.argv[1] in set(['TransE', 'Transe', 'transe']):
		model = 'transe'
		return [model, bi_diagonal] 
  
	if sys.argv[1] in set(['Bilinear', 'bilinear']):
		model = 'bilinear'
		if len(sys.argv) > 2:
			if sys.argv[2] in set(['Diagonal', 'diagonal', 'Diag', 'diag']):
				bi_diagonal = True 
			if sys.argv[2] in set(['Decomposed', 'decomposed', 'Decomp', 'decomp']):
				model = 'decomposed'
		return [model, bi_diagonal] 
	
	
	else: 
		print "\n\n   Type python kbc_main.py ...\n"
		print '   ... transe\n   ... bilinear [diagonal, decomposed]\n\n'
		return 



def select_KBC_model(model_tags): 

 	model = model_tags[0]
	bi_diagonal = model_tags[1]

	if model == 'transe':
		from TransE import TransE
		KBC_Model = TransE(params.dataset, params.swap, params.dim, params.margin, params.l1_flag, params.device, params.memory, params.learning_rate, params.max_epoch, params.batch_size, params.test_size, params.result_log_cycle, eval_with_np=params.eval_with_np, shuffle_data=params.shuffle_data, check_collision=params.check_collision, normalize_ent=params.normalize_ent)
		return KBC_Model
		
	if model=='bilinear': 
		from Bilinear import Bilinear
		KBC_Model = Bilinear(params.dataset, params.swap, params.dim, params.margin, params.device, params.memory, params.learning_rate, params.max_epoch, params.batch_size, params.test_size, params.result_log_cycle, diagonal=bi_diagonal, eval_with_np=params.eval_with_np, shuffle_data=params.shuffle_data, check_collision=params.check_collision, normalize_ent=params.normalize_ent, dropout=params.dropout)
		return KBC_Model

 	if model=='decomposed': 
		from Bilinear_Decomp import Bilinear_Decomp
	    	KBC_Model = Bilinear_Decomp(params.dataset, params.swap, params.dim, params.dim_hidden, params.margin, params.device, params.memory,  params.learning_rate, params.max_epoch, params.batch_size, params.test_size, params.result_log_cycle, eval_with_np=params.eval_with_np,  shuffle_data=params.shuffle_data, check_collision=params.check_collision, normalize_ent=params.normalize_ent, dropout=params.dropout)
		return KBC_Model



def kbc_main():

    model_tags = process_sys_argv()

    if model_tags == None: 
    	return 

    KBC_Model = select_KBC_model(model_tags)

    eval_mode, filtered = KBC_Model.get_eval_tags(sys.argv)

    PATHS, PLOT_PATHS = KBC_Model.getPATHS()

    # load set of all triples, train, valid and test data
    triples, train, valid, test = KBC_Model.load_data()

    # load dicts (URI strings to int) 
    ent_URI_to_int, rel_URI_to_int = KBC_Model.create_dicts(triples)    

    # load input-formats for script: triples_set (for faster existential checks) and int-matrices for triple stores (train-, test- and valid-set)
    triples_set, train_matrix, valid_matrix, test_matrix  = KBC_Model.create_int_matrices(triples, train, valid, test, ent_URI_to_int, rel_URI_to_int)

    # entity and relation-lists:
    n = len(ent_URI_to_int) #number of all unique entities
    m = len(rel_URI_to_int) #number of all unique relations

    KBC_Model.run_training(PATHS, PLOT_PATHS, n, m, eval_mode, filtered, triples_set, train_matrix, valid_matrix, test_matrix)

	 


def main(arg=None):
    kbc_main()
    
if __name__=="__main__": 
    #tf.app.run()  # allows a TF-flag-passthrough 
    main()  

