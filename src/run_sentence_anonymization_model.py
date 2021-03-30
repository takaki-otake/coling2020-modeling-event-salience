import os
import sys
import torch
import tqdm
import datetime
import argparse
import pandas as pd
from glob import glob
import logzero
from logzero import logger
from itertools import islice
from more_itertools import flatten
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

#important for reproducible results
torch.random.manual_seed(0)

def extract_ith_contexts(indexed_tokens, ith, max_context_len):
	len_list = [len(tokenized_sent) for tokenized_sent in indexed_tokens]
	target_sent_len = len_list[ith - 1]

	full_context_before_len = sum(len_list[:ith - 1])
	full_context_after_len = sum(len_list[ith:])

	tmp_context_before_len = (max_context_len - target_sent_len) * 0.5
	tmp_context_after_len = (max_context_len - target_sent_len) * 0.5

	if sum(len_list) <= max_context_len:  # full_context_before_len <= context_before_len and full_context_after_len <= context_after_len
		max_context_before_len, max_context_after_len = full_context_before_len, full_context_after_len
	elif full_context_before_len <= tmp_context_before_len and full_context_after_len > tmp_context_after_len:
		max_context_before_len = full_context_before_len
		max_context_after_len = max_context_len - (full_context_before_len + target_sent_len)
	elif full_context_before_len > tmp_context_before_len and full_context_after_len <= tmp_context_after_len:
		max_context_after_len = full_context_after_len
		max_context_before_len = max_context_len - (full_context_after_len + target_sent_len)
	elif full_context_before_len > tmp_context_before_len and full_context_after_len > tmp_context_after_len:
		max_context_before_len = tmp_context_before_len
		max_context_after_len = tmp_context_after_len
	else:
		print("you wrote wrong if statement")
		sys.exit(1)

	# extract context before
	context_before = []
	context_before_len = []
	for tokenized_sent, sent_len in zip(indexed_tokens[:ith - 1][::-1], len_list[:ith - 1][::-1]):
		if sum(context_before_len) + sent_len <= max_context_before_len:
			context_before = context_before + tokenized_sent[::-1]
			context_before_len.append(sent_len)

	# extract context after
	context_after = []
	context_after_len = []
	for tokenized_sent, sent_len in zip(indexed_tokens[ith:], len_list[ith:]):
		if sum(context_after_len) + sent_len <= max_context_after_len:
			context_after = context_after + tokenized_sent
			context_after_len.append(sent_len)

	context_before.reverse()

	#print("Processing {}th sentence".format(ith))
	#print("full context len: {}, {}".format(full_context_before_len, full_context_after_len))
	#print("max context len: {}, {}".format(max_context_before_len, max_context_after_len))
	#print("context len: {}, {}".format(len(context_before), len(context_after)))
	#print()
	return context_before, context_after


def compute_prob_for_rest_of_story(model, device, indexed_tokens_original, indexed_tokens_anonimized, ith, max_context_len, method):

	context_before, context_after = extract_ith_contexts(indexed_tokens_original, ith, max_context_len)
	context_after_len = len(context_after)

	# include target sentence
	input_tensor_w_sk = torch.tensor(context_before + indexed_tokens_original[ith - 1] + context_after).unsqueeze(0)

	if method == "VA" or method == "PAA":
		# target sentence with anonimized events
		input_tensor_wo_sk = torch.tensor(context_before + indexed_tokens_anonimized[ith - 1] + context_after).unsqueeze(0)
	elif method == "SD":
		# does not include target sentence
		input_tensor_wo_sk = torch.tensor(context_before + context_after).unsqueeze(0)
	else:
		print("Invalid method name")
		sys.exit(1)

	if device >= 0:
		device_name = 'cuda:{}'.format(device)
		input_tensor_w_sk = input_tensor_w_sk.to(device_name)
		input_tensor_wo_sk = input_tensor_wo_sk.to(device_name)
	else:
		pass


	with torch.no_grad():
		outputs = model(input_tensor_w_sk)

		logits, *_ = outputs

		# Shift so that tokens < n predict n
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = input_tensor_w_sk[..., 1:].contiguous()

		loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
		loss_w_sk = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

	with torch.no_grad():
		outputs = model(input_tensor_wo_sk)
		logits, *_ = outputs

		# Shift so that tokens < n predict n
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = input_tensor_wo_sk[..., 1:].contiguous()

		loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
		loss_wo_sk = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

	#print(loss_w_sk.size()[0], input_tensor_w_sk.size()[1])
	assert loss_w_sk.size()[0] == input_tensor_w_sk.size()[1] - 1, "loss_len: {}, input_len: {}".format(loss_w_sk.size()[0], input_tensor_w_sk.size()[1])
	assert loss_wo_sk.size()[0] == input_tensor_wo_sk.size()[1] - 1, "loss_len: {}, input_len: {}".format(loss_wo_sk.size()[0], input_tensor_wo_sk.size()[1])

	unnormalized_prob_w_sk = loss_w_sk[-context_after_len:].sum().item()
	unnormalized_prob_wo_sk = loss_wo_sk[-context_after_len:].sum().item()

	normalized_prob_w_sk = loss_w_sk[-context_after_len:].sum().item() / context_after_len
	normalized_prob_wo_sk = loss_wo_sk[-context_after_len:].sum().item() / context_after_len

	if args.normalization == "normalize":
		return normalized_prob_w_sk, normalized_prob_wo_sk
	elif args.normalization == "unnormalize":
		return unnormalized_prob_w_sk, unnormalized_prob_wo_sk
	else:
		print("invalid normalization method")
		sys.exit(1)

def check_file_alignment(file_path_list_1, file_path_list_2):
	for file_path_1, file_path_2 in zip(file_path_list_1, file_path_list_2):
		if os.path.basename(file_path_1) == os.path.basename(file_path_1):
			#logger.info("{} is aligned {}".format(file_path_1, file_path_2))
		else:
			logger.info("{} is misaligned {}".format(file_path_1, file_path_2))
			sys.exit(1)
	return None

def main(args):

	# logging all information with log level higher than 20(logging.INFO) to stdout
	logzero.loglevel(10)

	# preparing logging file path
	dt_now = datetime.datetime.now()

	logger.info('Preparing file path...')

	original_file_path_list = glob(os.path.normpath(args.input_original) + "/*")
	anonimized_file_path_list = glob(os.path.normpath(args.input_anonimized) + "/*")

	#check file pair alignment
	check_file_alignment(original_file_path_list, anonimized_file_path_list)

	if args.event_rem_method == "VA" or args.event_rem_method == "PAA":
		new_dir_path = os.path.normpath(args.output) + "/" + str(dt_now).replace(" ", "-") + "_" + args.event_rem_method + "_" + os.path.basename(args.model) + "_" + args.input_original.split("/")[-2] + "_" + args.input_anonimized.split("/")[-2] + "_" + args.normalization + "_main_bug_modified_fix_seed_context_eot"
		os.makedirs(new_dir_path)

		log_file_path = args.output + "logfile_" + str(dt_now).replace(" ", "-") + "_" + os.path.basename(args.model) + "_" + args.event_rem_method + "_" + args.input_original.split("/")[-2] + "_" + args.input_anonimized.split("/")[-2] + "_" + args.normalization + "_modified" + ".log"

		output_file_path_list = [
			new_dir_path + "/result_" + args.event_rem_method + str(dt_now).replace(" ", "-") + os.path.basename(
				args.model) + "_" + args.input_original.split("/")[-2] + "_" + args.input_anonimized.split("/")[-2] + "_" + args.normalization + "_modified" + "_" +
			os.path.basename(path).split(".")[
				0] + ".tsv" for path in original_file_path_list]

	elif args.event_rem_method == "SD":
		new_dir_path = args.output + str(dt_now).replace(" ", "-") + "_" + args.event_rem_method + os.path.basename(
			args.model) + "_" + args.input_original.split("/")[-2] + "_" + args.input_anonimized.split("/")[
						   -2] + "_" + args.normalization + "_modified"
		os.makedirs(new_dir_path)

		log_file_path = args.output + "logfile_" + str(dt_now).replace(" ", "-") + "_" + os.path.basename(
			args.model) + "_" + args.event_rem_method + "_" + args.input_original.split("/")[-2] + "_" +args.input_anonimized.split("/")[
							-2] + "_" + args.normalization + "_modified" + ".log"

		output_file_path_list = [
			new_dir_path + "/result_" + args.event_rem_method + str(dt_now).replace(" ", "-") + os.path.basename(
				args.model) + "_" + args.input_original.split("/")[-2] + "_" + args.input_anonimized.split("/")[-2] + "_" + args.normalization + + "_modified" + "_" +
			os.path.basename(path).split(".")[
				0] + ".tsv" for path in original_file_path_list]
	else:
		print("Invalid method name")
		sys.exit(1)

	# logging all information with log level higher than 10 (logging.DEBUG) to .log file
	logzero.logfile(log_file_path, loglevel=10)

	logger.info("Input args list: {}".format(args))
	logger.info('Loading model and tokenizer...')

	if args.model == "gpt2":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2')
	elif args.model == "gpt2-medium":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
	elif args.model == "gpt2-large":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2-large')
	elif args.model == "gpt2-xl":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
	elif args.model == "gpt":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
		# Load pre-trained model (weights)
		model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
	elif args.model == "distilgpt2":
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # distilgpt2 uses GTP-2 tokenizer
		model = GPT2LMHeadModel.from_pretrained('distilgpt2')
	elif args.model == "transformer-xl":
		tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
		model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
	else:
		tokenizer = GPT2Tokenizer.from_pretrained(args.model)
		model = GPT2LMHeadModel.from_pretrained(args.model)
		print("Load finetuned model from: {}".format(args.model))

	logger.info("Load pretrained model: {}".format(model))
	logger.info("Load pretrained tokenizer: {}".format(tokenizer))

	# add <unk> token to tokenizer
	#Note: Don't include this procedure when inputs text do not have <unk> token
	#This procedure leads to NOT reproducible results (because of using random initialised vector)
	# special_tokens_dict = {'unk_token': '<unk>'}
	# num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
	# logger.info('We have added' + str(num_added_tokens) + 'tokens')
	# model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
	# assert tokenizer.unk_token == '<unk>'

	# send to cuda if necessary
	if args.gpu >= 0:
		logger.info('Using GPU #{}'.format(args.gpu))
		device_name = 'cuda:{}'.format(args.gpu)
		model.to(device_name)


	# Set the model in evaluation mode to deactivate the DropOut modules
	# This is IMPORTANT to have reproducible results during evaluation!
	model.eval()

	max_context_len = args.contextlen
	logger.info("Max context len: {}".format(max_context_len))

	progress_bar1 = tqdm.tqdm(total=len(original_file_path_list))

	for (original_file_path, anonimized_file_path), output_file_path in zip(zip(original_file_path_list, anonimized_file_path_list), output_file_path_list):

		with open(original_file_path) as original_file:
			sentences_original = [line.strip().split("\t")[1] for line in original_file]

		with open(anonimized_file_path) as anonimized_file:
			sentences_anonimized = [line.strip().split("\t")[1] for line in anonimized_file]

		assert len(sentences_anonimized) == len(sentences_original), "loading invalid file set"

		indexed_tokens_original = [tokenizer.encode(sentence) for sentence in sentences_original]
		indexed_tokens_anonimized = [tokenizer.encode(sentence) for sentence in sentences_anonimized]

		assert indexed_tokens_original[-1] == indexed_tokens_anonimized[-1] == tokenizer.encode("<|endoftext|>")
		#assert not tokenizer.unk_token_id in set(flatten(indexed_tokens_original)), "<unk> token id exists in inputs original"
		#assert not tokenizer.unk_token_id in set(flatten(indexed_tokens_anonimized)), "<unk> token id exists in inputs modified"

		with open(original_file_path) as infile:

			progress_bar2 = tqdm.tqdm(total=len(indexed_tokens_original))

			result_list = []
			for ith, line in enumerate(infile.readlines()[:-1]): #exclude final sentence (i.e., <|endoftext|>)
				#with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens_original, indexed_tokens_anonimized, ith + 1, max_context_len, args.event_rem_method)

				if ith == len(sentences_original) - 2: #last sentence
					with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens_original, indexed_tokens_anonimized, ith + 1, max_context_len, args.event_rem_method)
				else:
					with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens_original[:-1], indexed_tokens_anonimized[:-1], ith + 1, max_context_len, args.event_rem_method)

				result_list.append([line.strip().split("\t")[0], line.strip().split("\t")[1], with_sk, without_sk, without_sk - with_sk])

				#logger.info("Original Sentence: " + sentences_original[ith] + ">>>" + "Anonimized Sentence: " + sentences_anonimized[ith])
				progress_bar2.update(1)

			df = pd.DataFrame(result_list, columns=["label", "sentence", "total_loss_with_sk", "total_loss_without_sk", "score"])
			df.to_csv(output_file_path, sep='\t')
			progress_bar1.update(1)

			logger.info("DONE: " + original_file_path + "&" + anonimized_file_path + ">>>" + output_file_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--event_rem_method', '-e', type=str, help='event removal method')
	parser.add_argument('--model', '-m', type=str, help='language model')
	parser.add_argument('--gpu', '-g', type=int, default=-1, help='use device')
	parser.add_argument('--normalization', '-n', type=str, help='normalization method')
	parser.add_argument('--contextlen', '-c', type=int, help='max context length')
	parser.add_argument('--input_original', '-io', type=str, help='directory path for original inputfiles')
	parser.add_argument('--input_anonimized', '-ia', type=str, help='directory path for anonimized inputfiles')
	parser.add_argument('--output', '-o', help='directory path for outputfiles')

	args = parser.parse_args()
	main(args)