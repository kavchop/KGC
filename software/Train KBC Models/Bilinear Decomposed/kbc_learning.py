import numpy as np
import tensorflow as tf
from Bilinear_Decomp import Bilinear_Decomp
import sys
import params



def kbc_learning(KBC_Class):

    KBC_Model = KBC_Class(params.dataset, params.swap, params.dim, params.dim_hidden, params.margin, params.device, params.learning_rate, params.max_epoch, params.batch_size, params.test_size, params.result_log_cycle, shuffle_data=params.shuffle_data, check_collision=params.check_collision, normalize_ent=params.normalize_ent)

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
    kbc_learning(Bilinear_Decomp)
    
if __name__=="__main__": 
    #tf.app.run()  # allows a TF-flag-passthrough 
    main()  

