import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from statistics import mean, variance, stdev
from sklearn.metrics import average_precision_score


def calculate_mean_score_diff(input_path_list):
	pos_score_list = []
	neg_score_list = []
	for file_path in input_path_list:
		df = pd.read_csv(file_path, sep='\t', engine='python')
		for _, series in df[["label", "score"]].iterrows():
			if series["label"] == 1:
				pos_score_list.append(series["score"])
			else:
				neg_score_list.append(series["score"])
	return sum(pos_score_list) / len(pos_score_list), sum(neg_score_list)/len(neg_score_list)

def calculate_AP_for_story(result_file_path):

	df = pd.read_csv(result_file_path, sep='\t', engine='python')

	gold_label = df["label"].astype(int)
	scores = df["score"].astype(float)
	ap = average_precision_score(gold_label, scores)
	spearman_corr = df[["label", "score"]].corr(method="spearman").iloc[0, 1]


	return ap, spearman_corr

def calculate_MeanAP(input_path_list):

	ap_score_list = []
	spearman_corr_list = []

	for result_file_path in input_path_list:
		ap_score, spearman_corr = calculate_AP_for_story(result_file_path)
		ap_score_list.append(ap_score)
		spearman_corr_list.append(spearman_corr)

	mean_AP = sum(ap_score_list) / len(ap_score_list)
	mean_spearman_corr = sum(spearman_corr_list) / len(spearman_corr_list)

	return mean_AP, mean_spearman_corr


def main(args):

	input_path_list = glob(os.path.normpath(args.input) + "/*")
	print("Process {} files".format(len(input_path_list)))
	print("Evaluation results from {} files".format(len(input_path_list)))
	print("Mean Average Precision (proposed): {}".format(calculate_MeanAP(input_path_list)[0]))
	
	#print("Average score for label 1: {}".format(calculate_mean_score_diff(input_path_list)[0]))
	#print("Average score for label 0: {}".format(calculate_mean_score_diff(input_path_list)[1]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', type=str, help='directory path for inputfiles')

	args = parser.parse_args()
	main(args)