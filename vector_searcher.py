import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import hashlib
from pysondb import db
from sklearn.neighbors import KDTree
import pickle
import os
import sys
import json

class VectorSearcher:
    
    def __init__(self,db_name, KDTree_name):
        self.model = None
        self.data_db = None
        self.db_name = db_name
        self.KDTree = None
        self.KDTree_name  = KDTree_name
        self.init_model()
        self.init_db()
        
    def init_model(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def init_db(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),self.db_name)
        self.data_db = db.getDb(path)
        
    def sentence_encoding_transform(self, to_encode):
        ''' to_encode should be a string or an array of strings'''
        
        if len(to_encode)<1:
            return "ERROR MY DUDE, CHECK ARRAY LENGTH"
        
        sentence_embeddings = self.model.encode(to_encode)
        sentence_embeddings = sentence_embeddings.astype(np.double)
        
        if len(to_encode) == 1 or type(to_encode)!=list:
            sentence_one_embedding = sentence_embeddings.reshape(1,sentence_embeddings.shape[0])
            return sentence_one_embedding
        else:
            sentence_many_embeddings = sentence_embeddings  
            return sentence_many_embeddings
        return
    
    
    #database stuff from here on out
    def structure_data_from_queries(self, queries:list, embedding_queries:list):
        ''' embedding_queries is the one that's to be masked
            queries is without the mask
            data is sql statement    
        '''
        sentence_embeddings = self.sentence_encoding_transform(embedding_queries)
        

        d = []
        for i,data in enumerate(zip(queries,embedding_queries, sentence_embeddings)):

            d.append({"index":i, "query":data[0], "embedding_query":data[1], "embedding":list(data[2]), "hash":hashlib.sha1(sentence_embeddings[i]).hexdigest(), "data":"```Dummy SQL```"})
    
        return d
    
    def check_exist(self, data):
        #does this hash exist?? 
        # hashlib.sha1(embedding).hexdigest()
        pass
    
    def add_to_db(self,d):
        #add data
        for i in d:
            qur = self.data_db.getByQuery(query={"index":i["index"]})
            if len(qur) !=0:

                self.data_db.updateByQuery(qur[0],i)
            else:
                self.data_db.add(i)
        
        print("added/updated")
        
    def add_one(self, data:dict):
        ''' Data should be in form 
        query is the actual query example, emberrding_query is the one using masks, and data is the code produced (SQL or Pandas or whatever yanno?)
        {query:xxx,embedding_query:xxx,data:xxx}''' 
        index = 0
        if len(self.data_db.getAll())!=0:
            index = self.data_db.getByQuery({"index":len(self.data_db.getAll()) - 1})[-1]["index"] + 1
        
        embedding = self.sentence_encoding_transform(data["embedding_query"])[0]
        payload = {"index":index,
                   "embedding_query":data["embedding_query"],
                   "query":data["query"], 
                   "embedding":list(embedding),
                   "data":data["data"], 
                   "hash":hashlib.sha1(embedding).hexdigest()}
        
        added = self.data_db.add(payload)
        
        self.build_and_save_KDTree( filename= self.KDTree_name)
        return added
    
    def get_embeddings(self):
        ret_embeddings = [x["embedding"] for x in sorted(self.data_db.getAll(), key=lambda x: x['index'])]
        return ret_embeddings
    #KD tree stuff
    
    
    def build_KDTree(self, embeddings):
        tree = KDTree(embeddings, leaf_size=10) 
        self.KDTree = tree
        # return tree

    def build_and_save_KDTree(self, filename= 'KDTree_ask.sav'):
        #build and save the tree that indexes 
        # if(filename not in ['KDTree_ask.sav', 
        #                     'KDTree_graph.sav',
        #                     'KDTree_predict.sav']):
        #     print("Error loading tree my dude")
        #     return None
        
        emb = self.get_embeddings()
        self.build_KDTree(emb)
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),filename)

        if self.KDTree:
            pickle.dump(self.KDTree, open(path, 'wb'))
        else:
            print("Yer object is empty my dude")

    def load_KDTree(self,filename= 'KDTree_ask.sav'):
        # if(filename not in ['KDTree_ask.sav', 
        #                     'KDTree_graph.sav',
        #                     'KDTree_predict.sav']):
        #     print("Error loading tree my dude")
        #     return None
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),filename)

        self.KDTree = pickle.load(open(path, 'rb'))
        
    def query_ind_KDTree(self, query, num_ne):
        # queries should be the query that you want to search in text
        query_transformed = self.sentence_encoding_transform(query)
        ind = self.KDTree.query(query_transformed, k=num_ne, return_distance=False)
        return ind[0]
    
    def get_data_from_index(self, arr):
        
        qur = [{"index":x} for x in arr]
        to_return = []
        for i in qur:
            to_return.append(self.data_db.getByQuery(i)[0])
        return to_return

    def get_data_from_query(self,query, num):
        #get data from a query, num is the number of example to search and pass through to the prompt
        idxs = self.query_ind_KDTree(query,num)
        return self.get_data_from_index(idxs)
