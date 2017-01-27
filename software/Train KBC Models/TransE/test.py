import os
import numpy as np
import zipfile

print os.getcwd()
print os.listdir(os.getcwd())
model_name = 'transe'
dim = 30
path = '../../data/Trained Models/'+model_name+'/dim = '+str(dim)
if not os.path.exists(path):
    os.makedirs(path)

#change directory to path
os.chdir('../../data/Triple Store/FB15K')

print os.listdir('.')

zip_ref = zipfile.ZipFile('data.zip', 'r')
zip_ref.extractall()
zip_ref.close()

TRAIN_FILE = 'valid.txt'
if os.path.isfile(TRAIN_FILE) and os.access(TRAIN_FILE, os.R_OK):
    print "True"
    train_triples = np.loadtxt(TRAIN_FILE,dtype=np.object,comments='#',delimiter=None)

    print train_triples[0:5]
else: 
    print "False"

'''
TRAIN_FILE = path = os.path.join(par, "/data/test.txt")			#os.chdir('../data/test.txt')
print TRAIN_FILE
print par
cur = os.getcwd()
print cur
TRAIN_FILE = path = os.path.join(cur, "/data/test.txt")
print TRAIN_FILE
TRAIN_FILE = '/data/train.txt'	
print TRAIN_FILE
print os.path.normpath(os.getcwd() + os.sep + os.pardir)
print os.listdir('./data')
TRAIN_FILE = './data/data.zip'

zip_ref = zipfile.ZipFile(TRAIN_FILE, 'r')
zip_ref.extractall('./data')
zip_ref.close()
print os.listdir('./data')

if os.path.isfile(TRAIN_FILE) and os.access(TRAIN_FILE, os.R_OK):
    print "True"
    train_triples = np.loadtxt(TRAIN_FILE,dtype=np.object,comments='#',delimiter=None)

    print train_triples
else: 
    print "False"
'''
