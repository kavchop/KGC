
import numpy as np 
import os 
import sys
sys.path.insert(0,'../')
from KBC_Util_Class import KBC_Util_Class


def triples_statistics(triples, dataset):
   
    # if triple store has redundant triples, remove first: 
    '''
    sorted_idx = np.lexsort(triples.T)
    sorted_data =  triples[sorted_idx,:]
    # Get unique row mask
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    # Get unique rows

    triples = sorted_data[row_mask]
    '''
	
    # unique relation list
    rel_list = list(set(triples[:,1]))

    # will create rel_map, a map from relation to relation_type (one_to_one etc)
    rel_map = {}
    
    for i in range(len(rel_list)):
        l = rel_list[i]
        triples_label  = triples[np.where(triples[:,1] == l)] #all triples with same relation
        num_heads = len(set(triples_label[:,0]))
        num_tails = len(set(triples_label[:,2]))
        
        if len(triples_label)/ float(num_heads) > 1.5: 
            rel_map[l] = 'one to '
        else:
            rel_map[l] = 'many to ' #e.g. 10/9 
        if len(triples_label)/ float(num_tails) > 1.5:
            rel_map[l] = rel_map[l] + 'one'
        else:
            rel_map[l] = rel_map[l] + 'many'
            

    # map from one_to_one (etc) to triples (with different relations)
    rel_type_to_triples = {} 
    for i in range(len(rel_list)):
        l = rel_list[i]
        #print 'rel', l, rel_map[l]
        if rel_map[l] not in rel_type_to_triples.keys(): 
            rel_type_to_triples[rel_map[l]] = triples[np.where(triples[:,1] == l)] 
            #print triples[np.where(triples[:,1] == l)] 
        else: 
            rel_type_to_triples[rel_map[l]] = np.append(rel_type_to_triples[rel_map[l]], triples[np.where(triples[:,1] == l)], axis= 0)


    print '\n**********Statistics on {} Triple Store**********\n'.format(dataset)

    e1 = triples[:,0]   #subject
    e2 = triples[:,2]   #object 
    rel = triples[:,1]  #predicate

    print '\nNumber of all triples in the knowledge base: {}'.format(len(triples))
    print '\nNumber of unique Entities: {}'.format(len(set(e1).union(set(e2))))
    print 'Number of unique Relations: {}\n'.format(len(set(rel))) 

    print '\nRelation-types between entities by triple count:\n'

    for rel_type in rel_type_to_triples:
	rel_type_triples = rel_type_to_triples[rel_type]

        print '{}: {} % - ({} / {})'.format(rel_type, 100 * round(len(rel_type_triples)/float(len(triples)), 4), len(rel_type_triples), len(triples))


    print '\nRelation-types between entities by relation count:\n'

    for rel_type in rel_type_to_triples:
	rel_type_triples = set(rel_type_to_triples[rel_type][:,1])

        print '{}: {} % - ({} / {})'.format(rel_type, 100 * round(len(rel_type_triples)/float(len(set(rel))), 4), len(rel_type_triples), len(set(rel)))


    print '\n'


def main(arg=None):

    dataset = 'Freebase'
    swap = True 
    Data = KBC_Util_Class(dataset, swap)
    if Data.data_exists(): 
        triples = Data.get_triple_matrix()
        triples_statistics(triples, dataset)


    dataset = 'Wordnet'
    swap = False 
    Data = KBC_Util_Class(dataset, swap)
    if Data.data_exists(): 
        triples = Data.get_triple_matrix()
        triples_statistics(triples, dataset)


    dataset = 'Cleanbase'
    swap = False 
    Data = KBC_Util_Class(dataset, swap)
    if Data.data_exists(): 
        triples = Data.get_triple_matrix()
        triples_statistics(triples, dataset)



if __name__=="__main__": 
    #tf.app.run()  # allows a TF-flag-passthrough 
    main()  
    
