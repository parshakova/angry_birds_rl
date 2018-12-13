import filter_env
from ddpg_off import *
import gc
from random import shuffle
import time, os
from operator import itemgetter

gc.enable()

EPISODES = 100000
TEST = 1


logfold = time.strftime("off%dd%mm_%H%M")
chkpt_dir = os.path.join(logfold,'chkpt')
if not os.path.exists(logfold):
    os.makedirs(logfold)
    os.makedirs(os.path.join(logfold,'chkpt'))

data_fname = 'all_trans13.pickle'

def add_padding(batch):
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

def main():    
    sess = tf.InteractiveSession()
    agent = DDPG(sess, data_fname)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, tf.train.latest_checkpoint('off13d06m_2002/chkpt'))
    summaries_var = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var.name)
            summaries_var.append(tf.summary.histogram(var.op.name, var))
    summary_op1 = tf.summary.merge(summaries_var)
    summary_writer1 = tf.summary.FileWriter(os.path.join(logfold,'summary_vars'), sess.graph)
#summary_op2 = tf.summary.scalar("loss", agent.critic_network.cost)
    summary_writer2 = tf.summary.FileWriter(os.path.join(logfold,'loss'), sess.graph)

    n_batches = len(agent.replay_buffer.buffer) // Hp.batch_size
    indices = range(n_batches*Hp.batch_size)
    for ep in xrange(Hp.epochs):
        shuffle(indices)
        for b in xrange(n_batches):
            sub_inds = indices[b*Hp.batch_size:(b+1)*Hp.batch_size]
            batch_el = list(itemgetter(*sub_inds)(agent.replay_buffer.buffer))
            pad_batch = add_padding(batch_el)

            cost = agent.train_off(pad_batch)

            if b %50 == 0:
                print("%d iter in ep = %d critic loss is %.4f " %(b,ep, cost))
                checkpoint_path = os.path.join(chkpt_dir,'model%d.ckpt'%(ep*n_batches+b))
                saver.save(sess, checkpoint_path, global_step=ep*n_batches+b)

            if b % 30 == 0:
#print('summary')
                summary_str1 = sess.run(summary_op1)
                summary_writer1.add_summary(summary_str1, ep*n_batches+b)
                summary_str2 = agent.summary_str2
                summary_writer2.add_summary(summary_str2, ep*n_batches+b)


if __name__ == '__main__':
    main()
