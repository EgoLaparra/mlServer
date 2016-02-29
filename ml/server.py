'''
Created on 4 Feb 2016

@author: egoitz
'''
# saved as greeting-server.py
import Pyro4
import crf.chainCRF as crf

class mlServer(object):
    def crfTrain(self, dataset_X, dataset_Y, file_base):
        crf.train(dataset_X, dataset_Y, file_base)
        del(dataset_X)
        del(dataset_Y)
    def crfTag(self, dataset_X, file_base):
        return crf.tag(dataset_X, file_base)
        del(dataset_X)


daemon = Pyro4.Daemon()                # make a Pyro daemon
ns = Pyro4.locateNS()                  # find the name server
uri = daemon.register(mlServer)   # register the greeting maker as a Pyro object
ns.register("mlServer", uri)   # register the object with a name in the name server

print("Ready.")
daemon.requestLoop()                   # start the event loop of the server to wait for calls
