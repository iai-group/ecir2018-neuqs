from collections import defaultdict
import numpy as np
import sys


def read_trec_gt(path):
	""" read Trec ground truth file to get a dictionary 
	{q1:[s1, s2, ...], q2:[s, ...], ....}
	where q is query, [s1, s2, ...] are relevant suggestions
	"""
	relevance_gt_dict = defaultdict(list)
	with open(path, "r", encoding="latin1") as f:
		for line in f:
			query, _, suggestion, _, r, _ = line.strip().split("\t")
			if r == "1" or r == "2":
				relevance_gt_dict[query].append(suggestion)
	return relevance_gt_dict


def read_gt(path):
	""" read ground truth file to get a dictionary 
	{q1:[s1, s2, ...], q2:[s, ...], ....}
	where q is query, [s1, s2, ...] are relevant suggestions
	"""
	relevance_gt_dict = defaultdict(list)
	with open(path, "r", encoding="latin1") as f:
		for line in f:
			query, _, suggestion, r = line.strip().split("\t")
			if r == "1":
				relevance_gt_dict[query].append(suggestion)
	return relevance_gt_dict


def read_run(path):
	""" read run file to get a dictionary 
	{q1:[s1, s2, ...], q2:[s, ...], ....}
	where q is query, [s1, s2, ...] are ranked suggestions
	"""
	run_dict = defaultdict(list)
	with open(path, "r", encoding="latin1") as f:
		for line in f:
			query, _, suggestion, _, _, _ = line.strip().split("\t")
			run_dict[query].append(suggestion)
	return run_dict


def cal_precision_at_k(rel_suggestions, gen_suggestions, k):
	"""given relevant suggestions and candidate suggestions for a query,
	calculate precision at k.
	"""
	# precisions a list of precisions, the ith element (precisions[i]) 
	# is the precision at (i+1)-th position
	# rel_count counting the number of relevant candidate suggestions
	precision, rel_count = 0, 0
	k_gen_suggestions = gen_suggestions[:k]

	# iter all candidate suggestions
	for i in range(len(k_gen_suggestions)):
		# get current suggestion
		gen_s = k_gen_suggestions[i]
		# if current suggestion appear in relevant suggestions
		if gen_s in rel_suggestions:
			# relevant suggestion count +1
			rel_count += 1.0

	precision = rel_count/k

	return precision


def eval_run(run_dict, relevance_gt_dict, run_name, k):
	""" evaluate each run
	
	- run_dict, a dictionray {q1:[s1, s2, ...], q2:[s, ...], ....}
	where q is query, [s1, s2, ...] are ranked suggestions
	
	- relevance_gt_dict, {q1:[s1, s2, ...], q2:[s, ...], ....}
	where q is query, [s1, s2, ...] are relevant suggestions in groundtruth
	
	- run_name, name of this run
	
	- k, precision at position k
	"""
	p_list, lens = [], []

	for query, results in run_dict.items():
		# evaluate the results of this query
		precision = cal_precision_at_k(relevance_gt_dict[query], results, k)
		l = len(results)
		
		p_list.append(precision)	
		lens.append(l)
	print("P@"+str(k)+":\t"+str(np.sum(p_list)/100)+"\t"+run_name)
	# print("P@"+str(k)+":\t"+str(np.sum(p_list)/100)+"\t"+str(len(p_list))+"\t"+run_name)


if __name__ == '__main__':
	k = int(sys.argv[1])
	relevance_gt_dict = read_gt(path="./data/QAC_ground_truth.tsv")

	run_pathes = ["./output/PopSuffix/QAC-run-AOL", "./output/PopSuffix/QAC-run-KnowHow", "./output/PopSuffix/QAC-run-WikiAnswer",
				  "./output/NLM/QAC-run-AOL", "./output/NLM/QAC-run-KnowHow", "./output/NLM/QAC-run-WikiAnswer",
				  "./output/Seq2Seq/QAC-run-AOL", "./output/Seq2Seq/QAC-run-KnowHow"]

	run_names = ["QAC-AOL-Suffix", "QAC-KnowHow-Suffix", "QAC-WikiAnswers-Suffix", 
				"QAC-AOL-NLM", "QAC-KnowHow-NLM", "QAC-WikiAnswers-NLM", 
				"QAC-AOL-Seq", "QAC-KnowHow-Seq"]

	for run_path, run_name in zip(run_pathes, run_names): 
		run_dict = read_run(path=run_path)
		eval_run(run_dict, relevance_gt_dict, run_name, k)

	# evaluate sigir17 task runs for QAC
	run_pathes = ["./output/KeyPhrase/QAC.runfile.txt", "./output/GoogleQS/QAC.runfile.txt"]
	run_names = ["QAC-KeyPhrase", "QAC-GoogleQS-SIGIR17"]

	for run_path, run_name in zip(run_pathes, run_names):
		run_dict = read_run(path=run_path)
		eval_run(run_dict, relevance_gt_dict, run_name, k)


	relevance_gt_dict = read_gt(path="./data/QR_ground_truth.tsv")

	run_pathes = ["./output/Seq2Seq/QR-run-AOL", "./output/Seq2Seq/QR-run-KnowHow"]
	run_names = ["QR-AOL-Seq", "QR-KnowHow-Seq"]

	for run_path, run_name in zip(run_pathes, run_names): 
		run_dict = read_run(path=run_path)
		eval_run(run_dict, relevance_gt_dict, run_name, k)

	# evaluate sigir17 task runs for QR
	run_pathes = ["./output/KeyPhrase/QR.runfile.txt", "./output/GoogleQS/QR.runfile.txt"]
	run_names = ["QR-KeyPhrase", "QR-GoogleQS-SIGIR17"]

	for run_path, run_name in zip(run_pathes, run_names): 
		run_dict = read_run(path=run_path)
		eval_run(run_dict, relevance_gt_dict, run_name, k)
