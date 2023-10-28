'''
Short Text Topic Modeling via DRTM
'''
import time
import numpy as np
from numpy.linalg import norm
import codecs

class DRTM(object):
    def __init__(
        self,
        D, A, B, S,
        IW=[], IC=[], IH=[],
        alpha=1.0, beta=0.1, miu=1.0, n_topic=10, max_iter=100, max_err=1e-3, 
        gamma=0.1, rho=0.5,
        fix_seed=False):
        '''
        
        '''
        for v in range(len(A)):# fz
            A[:,v] /= norm(A[:,v])
            B[:,v] /= norm(B[:,v])
            
            
        if fix_seed: 
            np.random.seed(0)
        # corpus, o_pmi, extra_info
        self.D = D
        self.S = S
        self.A = A
        self.B = B
        

        self.V = D.shape[0]
        self.n_D = D.shape[1]
        self.d = S.shape[0]#dimension

        self.n_topic = n_topic
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.miu = miu
        
        self.gamma= gamma
        self.rho = rho
        
        self.max_err = max_err

        
        self.nmf_init_rand()
        
        
        self.nmf_iter()

    def nmf_init_rand(self):
        self.W = np.random.random((self.V, self.n_topic)) # V*K  word-topic
        self.P = np.random.random((self.V, self.n_topic)) # V*K  relation-topic
        self.C = np.random.random((self.V, self.n_topic)) # V*K  context-topic
        self.H = np.random.random((self.n_D, self.n_topic))  # nD*K  doc-topic
        self.Q = np.random.random((self.d, self.n_topic))  # nD*K  doc-topic
        

        for k in range(self.n_topic):
            self.W[:, k] /= norm(self.W[:, k])
            self.P[:, k] /= norm(self.P[:, k])
            self.C[:, k] /= norm(self.C[:, k])
            self.H[:, k] /= norm(self.H[:, k])
            self.Q[:, k] /= norm(self.Q[:, k])


    def nmf_iter(self):
        loss_old = 1e20
        print('loop begin')
        start_time = time.time()
        '''
		core codes are coming soon
		'''

    def nmf_solver(self):
        '''
        using BCD framework ：block coordinate descent
        '''
        epss = 1e-20
        # Update W1 
        DH = np.dot(self.D, self.H)
        AC = np.dot(self.A, self.C)
        BP = np.dot(self.B, self.P)
        HtH = np.dot(self.H.T, self.H) #K*K
        CtC = np.dot(self.C.T, self.C)
        WtW = np.dot(self.W.T, self.W)
        PtP = np.dot(self.P.T, self.P)
        QtQ = np.dot(self.Q.T, self.Q)
        l1 = 0.5 * self.gamma *self.rho 
        l2 = 0.5 * self.gamma *(1-self.rho)
        # update W
        
        #Update H 
       
        # Update C
       
        # Update P
       
        #Update Q 
        '''
		core codes are coming soon
		'''
            
    def nmf_loss(self):

        loss = 0.5 * norm(self.D - np.dot(self.W, self.H.T), 'fro')**2 
        loss += 0.5 *  self.alpha*norm(np.dot(self.W, self.C.T)-self.A, 'fro')**2
        loss += 0.5 *  self.beta*norm(np.dot(self.W, self.P.T)-self.B, 'fro')**2
        loss += 0.5 *  self.miu*norm(np.dot(self.Q, self.H.T)-self.S, 'fro')**20
        
        loss += self.gamma * self.rho * ( norm(self.H, 1) + norm(self.W, 1))
        loss += 0.5 *  self.gamma * (1-self.rho) *  ( norm(self.H, 'fro')**2 + norm(self.W, 'fro')**2)
        
        return loss
    
    def get_lowrank_matrix(self):
        return self.W, self.C, self.H
    
    def save_format(self, Hfile='H.txt'):
        H = np.zeros_like(self.H)
        for i in range(np.shape(self.H)[0]):
            H[i] = self.H[i] / norm(self.H[i],1)
        np.savetxt(Hfile, H)
        
    def save_topics(self, Tfile='topic.txt', dictionary=None):
        n_topic = self.W.shape[1]
        topM = 30
        topics=[]
        for k in range(n_topic):
            temp_str= ['Topic {} : '.format(k)]
            for w in np.argsort(self.W[:,k])[::-1][:topM]:
                temp_str.append(dictionary[w])
            topics.append(' '.join(temp_str))
        
        with codecs.open(Tfile, 'w','utf-8') as fw:
            fw.writelines('\r'.join(topics))
            
        
