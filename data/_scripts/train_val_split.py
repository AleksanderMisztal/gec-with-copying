import json
import random

def orig_from_m2(m2_file):
	m2 = open(m2_file, encoding='utf-8').read().strip().split("\n\n")	
	return [sent.split('\n')[0][2:] for sent in m2]

# Apply the edits of a single annotator to generate the corrected sentences.
def corr_from_m2(m2_file, annotator_id=0):
	m2 = open(m2_file, encoding='utf').read().strip().split("\n\n")
	out = []
	# Do not apply edits with these error types
	skip = {"noop", "UNK", "Um"}
	
	for sent in m2:
		sent = sent.split("\n")
		cor_sent = sent[0].split()[1:] # Ignore "S "
		edits = sent[1:]
		offset = 0
		for edit in edits:
			edit = edit.split("|||")
			try:
				if edit[1] in skip: continue # Ignore certain edits
			except:
				print(m2_file, sent, edits)
			coder = int(edit[-1])
			if coder != annotator_id: continue # Ignore other coders
			span = edit[0].split()[1:] # Ignore "A "
			start = int(span[0])
			end = int(span[1])
			cor = edit[2].split()
			cor_sent[start+offset:end+offset] = cor
			offset = offset-(end-start)+len(cor)
		out.append(" ".join(cor_sent))
	return out

def process_set(name):
	orig = [sentence for path in paths[name] for sentence in orig_from_m2(path)]
	corr = [sentence for path in paths[name] for sentence in corr_from_m2(path)]
	assert len(orig) == len(corr)
	xys = [(x.strip(), y.strip()) for x, y in zip(orig, corr)]
	print(f"{name} set size: {len(xys)}")
	random.shuffle(xys)
	with open(f'./data/{name}.json', 'w', encoding='utf-8') as f:
		json.dump(xys, f, ensure_ascii=False, indent=2)


paths = {
	'train': [
		'./data/conll+nucle/nucle.train.gold.bea19.m2',
		'./data/fce/fce.train.gold.bea19.m2',
		'./data/wi+locness/ABC.train.gold.bea19.m2',
		'./data/fce/fce.dev.gold.bea19.m2',
	],
	'dev':[
		'./data/wi+locness/ABCN.dev.gold.bea19.m2', # 50 F_0.5 is really good 
	],
	'test': [
		'./data/conll+nucle/conll2014.2.test.gold.m2',
		'./data/fce/fce.test.gold.bea19.m2',
		#'./data/wi+locness/ABCN.test.bea19.orig'
	]
}

for name in paths.keys():
	process_set(name)