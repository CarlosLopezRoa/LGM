
# coding: utf-8

# In[61]:

import numpy as np
from operator import itemgetter


# In[77]:

class hmm:
    """
    General class that models an Hidden Markov Model
    It contains the triplet \lambda = \{\Pi, A, B\} here called: 
    """
    def __init__(self, stat_m, obs_m, stat_l, obs_l):
        """
        Constructor of the class
        """
        self.stat_m = stat_m
        self.obs_m = obs_m
        self.stat_l = stat_l
        self.obs_l = obs_l
        self.n_of_stat = len(stat_l)
        self.n_of_obs = len(obs_l) 
        self.stat_m_l = self.logs(self.stat_m)
        self.obs_m_l = self.logs(self.obs_m)
        
        
    def genseq (self, sequences = 1):
        """
        Generates the sequences
        """
        seq = []
        
        for iterator in np.arange(0,sequences):
            
            s = []
            o = []
            
            s.append(np.random.choice(len(self.stat_m[0]), 1, p=self.stat_m[0])[0])
            o.append(np.random.choice(len(self.obs_m[s[0]]), 1, p=self.obs_m[s[0]])[0])
            
            j = 1
            
            s_ = np.random.choice(len(self.stat_m[s[j - 1]]), 1, p=self.stat_m[s[j - 1]])[0]
            
            while s_ < len(self.stat_l) - 1 :
                
                s.append(s_)
                o.append(np.random.choice(len(self.obs_m[s_]), 1, p=self.obs_m[s_])[0])
                j = j + 1
                s_ = np.random.choice(len(self.stat_m[s[j - 1]]), 1, p=self.stat_m[s[j - 1]])[0]
            
            seq.append(itemgetter(*o)(self.obs_l))
        
        return seq
    
    def viterbi (self,obs):
        """
        
        """
        seq_ = []
        for iterator in np.arange(len(obs)):
            
            j = len(obs[iterator])
            d = np.zeros([self.n_of_stat, j])
            p = np.zeros([self.n_of_stat, j])
            obs_ = self.obs_l.index(obs[iterator][0])
            
            d[:,0] = self.stat_m_l[0, :] + self.obs_m_l[:, obs_]
            p[:, 0] = 0
            
            for j_ in np.arange(1,j):
                x_ = np.array([d[:, j_ - 1]]).T + self.stat_m_l
                obs_ = self.obs_l.index(obs[iterator][j_])
                d[:, j_] = np.amax(x_, axis = 0) + self.obs_m_l[:, obs_]
                p[:, j_] = np.argmax(x_, axis = 0)
            
            m_ = np.exp(np.max(d[:, j - 1]))
            seq = np.array([np.argmax(d[:, j - 1])], dtype = 'int')
            
            for j_ in np.arange(j - 1, 0, -1):
                seq = np.insert(seq, 0, p[seq[0], j_])
            
            seq_.append((seq, m_))
            
        return seq_
    
    def baum_welch(self, obs_, max_ = 30):

        logs = []

        for j_ in np.arange(max_):

            log_ = 0
            b_ = np.zeros([self.n_of_stat])
            a_ = np.zeros([self.n_of_stat])
            a__ = np.zeros([self.n_of_stat, self.n_of_stat])
            p_ = np.zeros([self.n_of_stat])
            b__ = np.zeros([self.n_of_stat, self.n_of_obs])

            for obs in obs_:

                a, l_, c = self.fwd(obs)
                b = self.bwd(obs, c)
                log_ += l_
                T = len(obs)
                w_k = 1.0 / -(l_ + np.log(T))
                g_ = a * b
                g = g_ / g_.sum(0)
                p_ += w_k * g[:, 0]
                b_ += w_k * g.sum(1)
                a_ += w_k * g[:, :T - 1].sum(1)
                xi = np.zeros([self.n_of_stat, self.n_of_stat, T - 1])

                for t in np.arange(T - 1):

                    obs__ = self.obs_l.index(obs[t + 1])

                    for i in np.arange(self.n_of_stat):

                        xi[i, :, t] = a[i, t] * self.stat_m[i, :] * self.obs_m[:, obs__] * b[:, t + 1]

                a__ += w_k * xi[:, :, :T - 1].sum(2)
                obs_m_ = np.zeros([self.n_of_stat, self.n_of_obs])

                for k in np.arange(self.n_of_obs):

                    i_ = np.array([self.obs_l[k] == x for x in obs])
                    obs_m_[:, k] = g.T[i_, :].sum(0)

                b__ += w_k * obs_m_

            stat_m_ = np.zeros([self.n_of_stat, self.n_of_stat])
            stat_m_[0, :] = p_ / np.sum(p_)

            for i in np.arange(1, self.n_of_stat - 1):
                stat_m_[i, :] = a__[i, :] / a_[i]

            self.stat_m = stat_m_

            for i in np.arange(self.n_of_stat):

                if b_[i] > 0:

                    b__[i, :] = b__[i, :] / b_[i]

                else:

                    b__[i, :] = b__[i, :]
            self.obs_m = b__
            logs.append(log_)
            if j_ > 1 and logs[j_ - 1] == log_:
                break
                
        return self
    
        
    def fwd (self, obs):
        
        j = len(obs)
        c = np.zeros(j)
        a = np.zeros([self.n_of_stat,j])
        obs_ = self.obs_l.index(obs[0])
        a[:, 0] = self.stat_m[0, :] * self.obs_m[:, obs_]
        c[0] = 1.0 / np.sum(a[:, 0])
        a[:, 0] *= c[0]
        
        for j_ in np.arange(1,j):
            obs_ = self.obs_l.index(obs[j_])
            a[:, j_] = np.dot(a[: , j_ - 1], self.stat_m) * self.obs_m[:, obs_]
            c[j_] = 1.0 / np.sum(a[:, j_])
            a[:, j_] *= c[j_]
        
        log_o = -(np.sum(np.log(c)))
        
        return a, log_o, c
    
    def bwd (self, obs, c):
        
        j = len(obs)
        b = np.zeros([self.n_of_stat, j])
        b[:, j - 1] = c[j - 1]
        
        for j_ in np.arange(j - 1, 0, -1):
            obs_ = self.obs_l.index(obs[j_])
            b[:, j_ - 1] = np.dot(self.stat_m, self.obs_m[:, obs_] * b[:, j_])
            b[:, j_ - 1] *= c[j_ - 1]
            
        return b
            
    def logs(self, log):
        log_ = np.zeros((log.shape))
        log_[log > 0] = np.log(log[log > 0])
        log_[log == 0] = float('-inf')
        return log_


# In[78]:

labels = ['INIT','Onset','Mid','End','FINAL']

obs = ['C1','C2','C3','C4','C5','C6','C7']

A = np.array([[0,1,0,0,0],[0,0.3,0.7,0,0],[0,0,0.9,0.1,0],[0,0,0,0.4,0.6],[0,0,0,0,0]])

B = np.array([[0,0,0,0,0,0,0],[0.5,0.2,0.3,0,0,0,0],[0,0,0.2,0.7,0.1,0,0],[0,0,0,0.1,0,0.5,0.4],[0,0,0,0,0,0,0]])

stat_m0 = np.array([[0, 1, 0, 0, 0],[0, 0.5, 0.5, 0, 0],[0, 0, 0.5, 0.5, 0],[0, 0, 0, 0.5, 0.5],[0, 0, 0, 0, 0]])

obs_m0 = np.array([[0, 0, 0, 0, 0, 0, 0],[0.30, 0.30, 0.40, 0, 0, 0, 0],[0, 0, 0.30, 0.30, 0.40, 0, 0],[0, 0, 0, 0.30, 0, 0.30, 0.40],[0, 0, 0, 0, 0, 0, 0]])

seq1 = [['C1','C2','C3','C4','C4','C6','C7'],['C2','C2','C5','C4','C4','C6','C6']]


# In[79]:

hmm1 = hmm(stat_m0, obs_m0, labels, obs)


# In[86]:

hmm1.viterbi(seq1)


# In[85]:

hmm1.baum_welch(seq1).stat_m


# In[82]:

hmm1.viterbi(seq1)


# In[ ]:




# In[ ]:



