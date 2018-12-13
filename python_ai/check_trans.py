import pickle 

res_f = 'all_trans.pickle'

with open(res_f, 'rb') as f:
	li = pickle.load(f) # or however you load the file
print("initial length %d "%len(li))

result = []

for tran in li:
	if len(tran) ==5:
		#(state, action, reward, new_state, done)
		if len(tran[0])==7 and len(tran[3])==7 and len(tran[1])==3:
			result.append(tran)


print("LENGTH OF BUFFER %d" % len(result))

with open(res_f,'wb') as wfp:
	pickle.dump(result, wfp)


