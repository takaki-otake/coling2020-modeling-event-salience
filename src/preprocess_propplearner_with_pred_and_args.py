import re
import os
import sys
sys.path.append(".")
import argparse
from tqdm import tqdm
from glob import glob
from itertools import islice
from pprint import pprint
from propplearnerutils import ProppLearnerDocument

be_set = {"am", "is", "are", "was", "were"}
verbs = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
valid_args = {"ARG0", "ARG1"}

def remove_sucsessive_tokens(token_list, target_token_id_set):
    previous_token=""
    result_token_list=[]
    
    for token_id in token_list:
        if token_id in target_token_id_set and token_id == previous_token:
            pass
        else:
            result_token_list.append(token_id)
        previous_token = token_id
    return result_token_list

def main(args):
    
    original_file_path_list = glob(os.path.normpath(args.input) + "/*")
    
    doc_list = []
    for file_path in original_file_path_list:
        document_inst = ProppLearnerDocument(file_path)
        document_inst.set_all_annotation_info()
        doc_list.append(document_inst)
    
    if not os.path.exist(args.output):
        os.mkdir(args.output)
    
    output_file_path_list = []
    for i in original_file_path_list:
        output_file_path_list.append(os.path.normpath(args.output) + "/" + os.path.basename(i).replace(".sty", ".tsv"))
    
    for original_file_path, output_file_path in tqdm(zip(original_file_path_list, output_file_path_list)):
        document_inst = ProppLearnerDocument(original_file_path)
        document_inst.set_all_annotation_info()

        #document_inst.add_special_token("<unk>",-1)
        document_inst.add_special_token("someone",-1)
        document_inst.add_special_token("something",-8)

        document_inst.add_special_token("be",-2)
        document_inst.add_special_token("do",-3)
        document_inst.add_special_token("did",-4)
        document_inst.add_special_token("doing",-5)
        document_inst.add_special_token("done",-6)
        document_inst.add_special_token("does",-7)

        with open(output_file_path, "w") as outfile:
            for line_num in document_inst.order_preserved_sentences:
                result_token_list = []
                sentence_inst = document_inst.sentences[line_num]
                should_be_replaced_tokens_arg0 = []
                should_be_replaced_tokens_other = []

                for srl in sentence_inst.srllist:
                    for arg_type, tokenids in srl.argdict.items():
                        if arg_type == "ARG0":
                            should_be_replaced_tokens_arg0 += tokenids
                        elif arg_type in valid_args:
                            should_be_replaced_tokens_other += tokenids
                        else:
                            pass

                for token_id in sentence_inst.tokens:
                    current_token_id = token_id
                    token_inst = document_inst.tokens[token_id]
                    
                    if token_id in should_be_replaced_tokens_arg0:
                        current_token_id = -1
                    elif token_id in should_be_replaced_tokens_other:
                        current_token_id = -8
                    else:
                        pass

                    if token_inst.pos in verbs:
                        if token_inst.token in be_set:
                            current_token_id = -2
                        elif token_inst.pos == "MD": 
                            pass
                        elif token_inst.pos == "VB": 
                            current_token_id = -3
                        elif token_inst.pos == "VBD": 
                            current_token_id = -4
                        elif token_inst.pos == "VBG": 
                            current_token_id = -5
                        elif token_inst.pos == "VBN": 
                            current_token_id = -6
                        elif token_inst.pos == "VBP": 
                            current_token_id = -3
                        elif token_inst.pos == "VBZ": 
                            current_token_id = -7
                        else:
                            raise ValueError
                            print("Unspecified Verb Type: {}".format(token_inst.token))
                    else:
                        pass

                    result_token_list.append(current_token_id)

                if sentence_inst.functions:
                    print("1", end="\t", file=outfile)
                else:
                    print("0", end="\t", file=outfile)

                #remove sucsessive (replaced) tokens
                result_token_list = remove_sucsessive_tokens(result_token_list, {-8, -1})
                print(" ".join([document_inst.tokens[token_id].token for token_id in result_token_list]), file=outfile)

            print("0\t", end="", file=outfile)
            print("<|endoftext|>", end="", file=outfile)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='path to the directory which contains raw propplearner .xml files')
    parser.add_argument('--output', '-o', help='path to the directory for processed files')
    args = parser.parse_args()
    
    main(args)