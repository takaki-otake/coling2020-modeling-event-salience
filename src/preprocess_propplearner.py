import re
import os
import argparse
from glob import glob
import xml.etree.ElementTree as ET


def get_sent_num(token_id, sent_identifier):
    sent_num = -1
    for key, value in sent_identifier.items():
        if token_id in value:
            sent_num = key
    return sent_num

def main(args):
    original_file_path_list = glob(os.path.normpath(args.input) + "/*")
    
    os.mkdir(args.output)
    
    output_file_path_list = []
    for i in original_file_path_list:
        output_file_path_list.append(os.path.normpath(args.output) + "/" +os.path.basename(i).replace(".sty", ".tsv"))

    for original_file_path, output_file_path in zip(original_file_path_list, output_file_path_list):

        tree = ET.parse(original_file_path)
        root = tree.getroot()
        text = root.findall(".//*[@id='edu.mit.story.char']/desc")[0].text

        id_to_token = {}
        for line in root.findall(".//*[@id='edu.mit.parsing.token']/desc"):
            token = line.text
            token_offset = int(line.get("off"))
            token_len = int(line.get("len"))
            token_id = line.get("id")
            
            if "`" in token or "'" in token:
                id_to_token[token_id] = token.replace("``", '"').replace("''", '"')
            else:
                id_to_token[token_id] = token

        assert len(id_to_token) == len(root.findall(".//*[@id='edu.mit.parsing.token']/desc"))

        order_preserved_sents = []

        sentid_to_tokenid = {}
        for line in root.findall(".//*[@id='edu.mit.parsing.sentence']/desc"):
            sent_offset = int(line.get("off"))
            sent_len = int(line.get("len"))
            sent_id = line.get("id")

            order_preserved_sents.append(sent_id)

            sentid_to_tokenid[sent_id] = line.text.split("~")

        assert len(sentid_to_tokenid) == len(root.findall(".//*[@id='edu.mit.parsing.sentence']/desc"))

        proppian_functions = {}
        for line in root.findall(".//*[@id='edu.mit.semantics.rep.function']/desc"):
            description = line.text.split("|")

            if description[0] == "alpha":
                pass
            else:
                statement_list = [re.split("[:,~]", item) for item in description[1:]]
                proppian_functions[description[0]] = [[get_sent_num(i, sentid_to_tokenid) for i in statement[1:]] if statement[0] == "ACTUAL" else [] for statement in statement_list]

        function_sentence_set = set()

        for state_lis in proppian_functions.values():
            for state in state_lis:
                function_sentence_set = function_sentence_set | set(state)

        with open(output_file_path, "w") as outfile:
            for sent_idx in order_preserved_sents:
                line = sentid_to_tokenid[str(sent_idx)]
                if sent_idx in function_sentence_set:
                    print("1\t", end="", file=outfile)
                    print(" ".join(id_to_token[token_id] for token_id in line), file=outfile)
                else:
                    print("0\t", end="", file=outfile)
                    print(" ".join(id_to_token[token_id] for token_id in line), file=outfile)
            print("0\t", end="", file=outfile)
            print("<|endoftext|>", end="", file=outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='path to the directory which contains raw propplearner .xml files')
    parser.add_argument('--output', '-o', help='path to the directory for processed files')
    args = parser.parse_args()
    
    main(args)