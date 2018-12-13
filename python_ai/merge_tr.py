import pickle 

d = dict()

file_l = ['trans_ben.pickle','trans_lec.pickle','trans_hin.pickle','trans_sch.pickle']
res_f = 'all_trans.pickle'

file_l = ['trans_sch1.pickle','trans_sch2.pickle']
res_f = 'trans_sch.pickle'

for file in file_l:
	
	try:
		2
		with open(file, 'rb') as f:
			li = pickle.load(f) # or however you load the file
	except EOFError:
		continue

	for trans in li:
		#(state, action, reward, new_state, done)
		pattern = "%.2f"
		floats = [pattern % i for i in trans[1]]
		key = str(trans[2]) +'_' +"".join(floats[:2])
		if not key in d:
			d[key] = trans

result = []

for pair in d.items():
	result.append(pair[1])

print("LENGTH OF BUFFER %d" % len(result))

with open(res_f,'wb') as wfp:
	pickle.dump(result, wfp)


