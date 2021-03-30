import re
import os
import spacy
import argparse
from glob import glob


def replace_event_in_sentence(sentence, spacy_model):

    be_set = {"am", "is", "are", "was", "were"}
    dep_label_set = {"nsubj", "aux", "expl"}

    def replace_verb_by_tag(token):
        if token.text in be_set:
            return "be"
        elif token.tag_ == "MD": 
            return token.text
        elif token.tag_ == "VB": 
            return "do"
        elif token.tag_ == "VBD": 
            return "did"
        elif token.tag_ == "VBG": 
            return "doing"
        elif token.tag_ == "VBN": 
            return "done"
        elif token.tag_ == "VBP": 
            return "do"
        elif token.tag_ == "VBZ": 
            return "does"
        else:
            print("Unspecified Verb Type: {}".format(token.text))
            return "do"

    def replace_token_sensitive(token):
        if token.pos_ == "VERB":
            return replace_verb_by_tag(token)
        else:
            return token.text

    def replace_token_sensitive_limit_dep(token):
        if token.pos_ == "VERB" and any([child.dep_ in dep_label_set for child in token.children]):
            return replace_verb_by_tag(token)
        else:
            return token.text

    def replace_token_dep(token):
        if token.pos_ == "VERB":
            children_dep_labels = set([child.dep_ for child in token.children])
            if "nsubj" in children_dep_labels:
                return "do"
            elif "aux" in children_dep_labels:
                return "do"
            elif "expl" in children_dep_labels:
                return "be"
            else:
                return token.text
        else:
            return token.text

    def replace_token(token):
        if token.pos_ == "VERB":
            return "do"
        else:
            return token.text

    parsed_sentence = spacy_model(sentence)

    #replaced_token_list = [replace_token_sensitive_limit_dep(token) for token in parsed_sentence]
    replaced_token_list = [replace_token_sensitive(token) for token in parsed_sentence]

    return " ".join(replaced_token_list)


def main(args):
    original_file_path_list = glob(os.path.normpath(args.input) + "/*")
    
    if not os.path.exist(args.output):
        os.mkdir(args.output)

    spacy_model = spacy.load("en_core_web_sm")
    
    output_file_path_list = []
    for i in original_file_path_list:
        output_file_path_list.append(os.path.normpath(args.output) + "/" + os.path.basename(i).replace(".sty", ".tsv"))

    for original_file_path, output_file_path in zip(original_file_path_list, output_file_path_list):
        with open(original_file_path) as infile, open(output_file_path, "w") as outfile:
            for line in infile:
                label, sentence = line.strip().split("\t")
                if sentence == "<|endoftext|>":
                    print(label, sentence, sep="\t", file=outfile)
                else:
                    replaced_sentence = replace_event_in_sentence(sentence, spacy_model)
                    print(label, replaced_sentence, sep="\t", file=outfile)

                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='path to the directory which contains raw propplearner .xml files')
    parser.add_argument('--output', '-o', help='path to the directory for processed files')
    args = parser.parse_args()
    
    main(args)