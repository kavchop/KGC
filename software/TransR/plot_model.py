import pickle
import numpy as np
import matplotlib.pyplot as plt



file_ = open('models/transr_results','r')
results = pickle.load(file_)    #list of tuples for pos_map and neg_map
file_.close()


print results 

results = np.array(results[1:len(results)], dtype=np.int32)
print results
#print int(results[2][1])+ int(results[1][1])

#a = np.reshape(np.array([[1,0,0,0,0]*len(results)]), (len(results),5))
#results =  results - a

epoch = results[:,0]
h_mean = results[:,1]
t_mean = results[:,2]

plt.ylabel('mean rank')
plt.xlabel('epoch')
#plt.axvline(x=230, linewidth=2, color='k')
plt.plot(epoch, h_mean, color='b', label='rank mean from head substitution')
plt.plot(epoch, t_mean, color='r', label='rank mean from tail substitution')
plt.title('Rescal - validation results during training')
plt.legend(loc='upper right')
plt.show()



