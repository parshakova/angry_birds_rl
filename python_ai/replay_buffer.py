from collections import deque
import random, pickle, os
import numpy as np
from tqdm import tqdm
from hp import Hp


class ReplayBuffer(object):
    def __init__(self, buffer_size, fname):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.load2(fname)
        # categories = redB, yelB, bluB, pig, ice, wood, stone 
    
    def load(self, fname):
       if os.path.exists(fname): 
          with open(fname,'rb') as rfp:
             scores = pickle.load(rfp)
             for tupl in scores:
                self.add(*tupl)

    def load2(self, fname):
        if os.path.exists(fname): 
            with open(fname,'rb') as rfp: 
                scores = pickle.load(rfp)
                for tupl in scores:
                    self.add(*tupl)
                # Here, All Samples and normalization should be done
                # Normalization each on : action, state, new_state
              
                reward = np.array(0)
                print(len(self.buffer))
                for e in tqdm(range(len(self.buffer))):
                  reward = np.append(reward,self.buffer[e][2])
                  for i in [0,3]:
                    if e == 0:
                      redb = self.buffer[e][i][0].copy()
                      yelb = self.buffer[e][i][1].copy()
                      blub = self.buffer[e][i][2].copy()
                      pig = self.buffer[e][i][3].copy()
                      ice = self.buffer[e][i][4].copy()
                      wood = self.buffer[e][i][5].copy()
                      stone = self.buffer[e][i][6].copy()
                    else:
                      redb = np.append(redb, self.buffer[e][i][0],axis=0)
                      yelb = np.append(yelb, self.buffer[e][i][1],axis=0)
                      blub = np.append(blub, self.buffer[e][i][2],axis=0)
                      pig = np.append(pig, self.buffer[e][i][3],axis=0)
                      ice = np.append(ice, self.buffer[e][i][4],axis=0)
                      wood = np.append(wood, self.buffer[e][i][5],axis=0)
                      stone = np.append(stone, self.buffer[e][i][6],axis=0)
                print(np.array(redb).shape)
                reward_norm = (np.mean(reward), np.std(reward))
                redb_norm = (np.mean(redb, axis=0), np.std(redb, axis=0))
                yelb_norm = (np.mean(yelb, axis=0), np.std(yelb, axis=0))
                blub_norm = (np.mean(blub, axis=0), np.std(blub, axis=0))
                pig_norm = (np.mean(pig, axis=0), np.std(pig, axis=0))
                ice_norm = (np.mean(ice, axis=0), np.std(ice, axis=0))
                wood_norm = (np.mean(wood, axis=0), np.std(wood, axis=0))
                stone_norm = (np.mean(stone, axis=0), np.std(stone, axis=0))
                
                redb_norm[1][4] = 0.0000001 if redb_norm[1][4]==0.0 else redb_norm[1][4]
                yelb_norm[1][4] = 0.0000001 if yelb_norm[1][4]==0.0 else yelb_norm[1][4]
                blub_norm[1][4] = 0.0000001 if blub_norm[1][4]==0.0 else blub_norm[1][4]
                pig_norm[1][4] = 0.0000001 if pig_norm[1][4]==0.0 else pig_norm[1][4]
                ice_norm[1][4] = 0.0000001 if ice_norm[1][4]==0.0 else ice_norm[1][4]
                wood_norm[1][4] = 0.0000001 if wood_norm[1][4]==0.0 else wood_norm[1][4]
                stone_norm[1][4] = 0.0000001 if stone_norm[1][4]==0.0 else stone_norm[1][4]
                
                print("original_buffer = ", self.buffer[e][i][0])
                print("redb_norm = ", redb_norm[0], " ", redb_norm[1])
                print("yelb_norm = ", yelb_norm[0], " ", yelb_norm[1])
                print("blub_norm = ", blub_norm[0], " ", blub_norm[1])
                print("pig_norm = ", pig_norm[0], " ", pig_norm[1])
                print("ice_norm = ", ice_norm[0], " ", ice_norm[1])
                print("wood_norm = ", wood_norm[0], " ", wood_norm[1])
                print("stone_norm = ", stone_norm[0], " ", stone_norm[1])
                
                print(redb_norm[0])
                for e in tqdm(range(len(self.buffer))):
                  self.buffer[e][2] = (self.buffer[e][2] - reward_norm[0])/reward_norm[1]
                  for i in [0,3]:
#print("e =", e," i = ",i ," redb ", redb_norm[0].shape, np.array(self.buffer[e][i][0]).shape, " pig ",pig_norm[0].shape,np.array(self.buffer[e][i][3]).shape)
                    self.buffer[e][i]= (np.divide(np.subtract(self.buffer[e][i][0], redb_norm[0]), redb_norm[1])[:,:-1], np.divide(np.subtract(self.buffer[e][i][1], yelb_norm[0]), yelb_norm[1])[:,:-1], np.divide(np.subtract(self.buffer[e][i][2], blub_norm[0]), blub_norm[1])[:,:-1],np.divide(np.subtract(self.buffer[e][i][3], pig_norm[0]), pig_norm[1])[:,:-1], np.divide(np.subtract(self.buffer[e][i][4], ice_norm[0]), ice_norm[1])[:,:-1], np.divide(np.subtract(self.buffer[e][i][5], wood_norm[0]), wood_norm[1])[:,:-1], np.divide(np.subtract(self.buffer[e][i][6], stone_norm[0]), stone_norm[1])[:,:-1])
               
    def save_tuples(self, fname):
        with open(fname,'wb') as wfp:
            pickle.dump(list(self.buffer), wfp)

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        batch = random.sample(self.buffer, batch_size)
        # list of tuples (state, action, reward, new_state, done)
        # where state is (redB, yelB, bluB, pig, ice, wood, stone) e.g. redB is (seqlen, n_coord) 
        def pad_state(batch, ind):
            max_lengths = [1]*Hp.categories
            for ar in batch:
                state = ar[ind] # s or s'
                for i in range(Hp.categories):
                    if state[i].shape[0] > max_lengths[i]:
                        max_lengths[i] = state[i].shape[0]

            for j in range(len(batch)):
                state = batch[j][ind]
                row = []
                for i in range(Hp.categories):
                    subl = np.zeros((max_lengths[i], Hp.n_coord))
                    #print(subl.shape, state[i].shape)
                    subl[:state[i].shape[0], :] = state[i]
                    row.append(subl)
                newrow = []
                for k in range(len(batch[j])):
                    if k == ind:
                        newrow += [row]
                    else:
                        newrow += [batch[j][k]]
                batch[j] = tuple(newrow)

        pad_state(batch, 0)
        pad_state(batch, 3)

        return batch

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        
        experience = [state, action, reward, new_state, done]
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0



"""
buffer = ReplayBuffer(3)

def trans_generator():
    g = lambda: 1 + np.random.randint(3)
    trans = [[np.random.rand(g(), Hp.n_coord)]*Hp.categories, np.random.rand(3), np.random.rand(1), [np.random.rand(g(), Hp.n_coord)]*Hp.categories, True]

    return trans

for j in range(8):
    val = trans_generator()
    buffer.add(*val)

l = buffer.get_batch(3)

"""
#b = ReplayBuffer(1000, "all_trans.pickle")
