'''
Created on 4 Feb 2016

@author: egoitz
'''

import numpy as np
import pickle as pk
from sklearn.feature_extraction import DictVectorizer
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM


def train (dataset_X, dataset_Y, file_base):       

    vect = DictVectorizer()
    node_list = []
    for sample in dataset_X:
        for node in sample:
            node_list.append(node)
    vectorized_node_list = vect.fit_transform(node_list).toarray()
    print (str(node_list.__sizeof__()) + " " + str(vectorized_node_list.__sizeof__()))
       
    del(node_list)
    vect_file = open(file_base + '.vector.pkl', 'wb')
    pk.dump(vect, vect_file)
    vect_file.close()
    del(vect)     
       
    X_train = []
    c = 0
    for i in range(0,len(dataset_X)):
        node = []
        for j in range(0,len(dataset_X[i])):
            node.append(vectorized_node_list[c])
            c+=1
        X_train.append(np.array(node))        
    X_train = np.array(X_train)
    del(vectorized_node_list)
    del(node)
    del(dataset_X)
    
    Y_train = []
    class_list = []
    for i in range(0, len(dataset_Y)):
        sample = []
        for j in range(0, len(dataset_Y[i])):
            if dataset_Y[i][j] not in class_list:
                class_list.append(dataset_Y[i][j])
            sample.append(class_list.index(dataset_Y[i][j]))
        Y_train.append(np.array(sample))   
    Y_train = np.array(Y_train)
    del(sample)
    del(dataset_Y)
    class_file = open(file_base + '.classes.pkl', 'wb')
    pk.dump(class_list, class_file)
    class_file.close()
    del(class_list)

    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model=model, C=1., max_iter=10, verbose=1)
    ssvm.fit(X_train, Y_train)
    del(X_train)
    del(Y_train)
    model_file = open(file_base + '.model.pkl', 'wb')
    pk.dump(ssvm, model_file)
    model_file.close()
    del(ssvm)

    


def tag (dataset_X, file_base):
    model_file = open(file_base + '.model.pkl', 'rb')
    ssvm = pk.load(model_file)
    model_file.close()
    vect_file = open(file_base + '.vector.pkl', 'rb')
    vect = pk.load(vect_file)
    vect_file.close()
    node_list = []
    for sample in dataset_X:
        for node in sample:
            node_list.append(node)
    vectorized_node_list = vect.transform(node_list).toarray()
    del(node_list)
    del(vect)
    
    X_train = []
    c = 0
    for i in range(0,len(dataset_X)):
        node = []
        for j in range(0,len(dataset_X[i])):
            node.append(vectorized_node_list[c])
            c+=1
        X_train.append(np.array(node)) 
    X_train = np.array(X_train)
    del(dataset_X)
    del(vectorized_node_list)
    
    prediction = ssvm.predict(X_train)
    del(ssvm)
    del(X_train)

    class_file = open(file_base + '.classes.pkl', 'rb')
    class_list = pk.load(class_file)
    class_file.close()
    prediction_out = []
    for sample in prediction:
        sample_out = []
        for node in sample:
            sample_out.append(class_list[node])
        prediction_out.append(sample_out)
    del(class_list)

    return prediction_out
    del(prediction_out)
    
  
