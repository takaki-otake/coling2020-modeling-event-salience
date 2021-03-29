import re
import spacy
from os import path
from glob import glob
import xml.etree.ElementTree as ET
from itertools import islice
from pprint import pprint
from typing import List
from sexpdata import loads, dumps
from collections import defaultdict
from nltk.tree import Tree, ParentedTree
from more_itertools import flatten


class ProppLearnerToken:
    def __init__(self, token, token_offset, token_len):
        self.token = token
        self.offset = token_offset
        self.len = token_len
        self.pos = ""
        
    def __repr__(self):
        return self.token

    
class ProppLearnerSentence:
    def __init__(self, text, sent_offset, sent_len):
        self.tokens = [int(i) for i in text.split("~")]
        self.offset = sent_offset
        self.len = sent_len
        self.sexpression = ""
        self.srllist = []
        self.functions = defaultdict(list)
        self.events = []
        
    def __repr__(self):
        return "ProppLearnerToken ID list: {}".format(self.tokens)

    
class ProppLearnerSRL:
    def __init__(self, srl_info, off_SRL, len_SRL, sentence):
        num_leaf, _, frame_id, _, *args = srl_info.split(" ")
        self.offset = int(off_SRL)
        self.len = int(len_SRL)
        self.num_leaf = int(num_leaf)
        #self.pred_token = ""
        self.frame_id = frame_id
        self.pred_token_id = None
        self.argdict = {}
        self.sentence_id = None
        
        def get_pred_token_id(leaf_n, sentence):
            tree = Tree.fromstring(sentence.sexpression)
            pred_leaf = tree.leaves()[leaf_n]
            leaf_str, token_id = pred_leaf.split("_")
            return int(token_id)
        
        def get_arg_token_ids(hight, child_idx, leaf, tree):

            new_tree = tree[tree.leaf_treeposition(child_idx)[:-1]]

            for i in range(hight):
                new_tree = new_tree.parent()

            token_id_list = []
            for l in new_tree.leaves():
                token, token_id = l.split("_")
                token_id_list.append(int(token_id))

            return token_id_list
        
        self.pred_token_id = get_pred_token_id(self.num_leaf, sentence)
        
        parented_tree = ParentedTree.fromstring(sentence.sexpression)
        re_obj = re.compile(r"(?P<span>.+?)-(?P<arg_label>.+?)-(?P<arg_feat>.*)")
        
        for arg in args:
            if ";" in arg:
                #print("expression -> ;")
                pass
            elif "," in arg:
                #print("expression -> ,")
                arg_label = re_obj.match(arg).group("arg_label")
                arg_feat = re_obj.match(arg).group("arg_feat")
                span_list = re_obj.match(arg).group("span").split(",")
                
                for span in span_list:
                    child_idx, hight = span.split(":")
                    leaf = parented_tree.leaves()[int(child_idx)]
                    leaf_token_id_list = get_arg_token_ids(int(hight), int(child_idx), leaf, parented_tree)
                    self.argdict[arg_label] = leaf_token_id_list
                    
            elif "*" in arg:
                #print("expression -> *")
                pass 
            else:
                child_idx, hight = re_obj.match(arg).group("span").split(":")
                arg_label = re_obj.match(arg).group("arg_label")
                arg_feat = re_obj.match(arg).group("arg_feat")
                
                leaf = parented_tree.leaves()[int(child_idx)]
                leaf_token_id_list = get_arg_token_ids(int(hight), int(child_idx), leaf, parented_tree)
                self.argdict[arg_label] = leaf_token_id_list
    
    def __repr__(self):
        return "Pred:{}. Args:{}".format(self.frame_id, self.argdict.items())
    
    
class ProppLearnerDocument:
    def __init__(self, original_xml_file_path):
        if path.exists(original_xml_file_path):
            self.file_path = original_xml_file_path
        else:
            print("File doesn't exist: {}".format(original_xml_file_path))
            
        self.title = original_xml_file_path.split("/")[-1].rstrip(".sty")
        self.text = ""
        self.tokens = {}
        self.event_tokens_list = []
        self.sentences = {}
        self.order_preserved_sentences = []
        self.sentspan2sentid = {}
        self.sentid2srls = defaultdict(list)
        self.tokenid2sentid = {}
        self.srls = {}
        self.lemmas = {}
        #self.parse_trees = {}
        #self.events = {}
        self.functions = {}
    
    def add_special_token(self, token, token_id):
        if token_id in self.tokens:
            print("Error: token id alredy exist")
        else:
            self.tokens[token_id] = ProppLearnerToken(token, -1, 0)
            print("Added new token:{}for id: {}".format(token, token_id))
    
    def set_all_annotation_info(self):
        
        def get_sent_id_from_token_id(token_id, sentences):
            for sentence_id, sentence in sentences.items():
                if token_id in sentence.tokens:
                    return sentence_id
            return None
    
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        
        self.text = root.findall(".//*[@id='edu.mit.story.char']/desc")[0].text
        
        #prosessing token annotaion infomation
        for line in root.findall(".//*[@id='edu.mit.parsing.token']/desc"):
            token = line.text
            token_offset = int(line.get("off"))
            token_len = int(line.get("len"))
            token_id = int(line.get("id"))

            if self.text[token_offset + 1:token_offset + token_len + 1] == token:
                self.tokens[token_id] = ProppLearnerToken(token, token_offset, token_len)
            else:
                token = token.replace("``", '"').replace("''", '"')
                self.tokens[token_id] = ProppLearnerToken(token, token_offset, token_len)
                
        #prosessing sentence annotation infomation
        for line in root.findall(".//*[@id='edu.mit.parsing.sentence']/desc"):
            sent_offset = int(line.get("off"))
            sent_len = int(line.get("len"))
            sent_id = int(line.get("id"))

            self.order_preserved_sentences.append(sent_id)
            self.sentences[sent_id] = ProppLearnerSentence(line.text, sent_offset, sent_len)    
            self.sentspan2sentid[(sent_offset, sent_len)] = sent_id
        
        for token_id, token in self.tokens.items():
            self.tokenid2sentid[token_id] = get_sent_id_from_token_id(token_id, self.sentences)
            
        #processing lemma annotation infoamation
        #for line in root.findall(".//*[@id='edu.mit.parsing.stem']/desc"):
            #off_lem = int(line.get("off"))
            #len_lem = int(line.get("len"))
            #id_lem = int(line.get("id"))

            #pos_annot_id, *lemma = line.text.split()
            
            #for token_id, token_instance in self.tokens.items():
                #if token_instance.offset == off_lem and token_instance.len == len_lem:
                    #token_instance.lemma = lemma
                    #self.lemmas.append(token_id)
                #else:
                    #pass
        
        #procession POS annotation infomation
        token_id_to_POS = {}
        for line in root.findall(".//*[@id='edu.mit.parsing.pos']/desc"):
            off_pos = int(line.get("off"))
            len_pos = int(line.get("len"))
            id_pos = int(line.get("id"))

            token_id, POS = line.text.split()
            token_id_to_POS[int(token_id)] = POS
        
        for token_id, token_instance in self.tokens.items():
            token_instance.pos = token_id_to_POS[token_id]
            
        
        #prosessing parsing annnotation infomation
        sentspan_to_sexp ={}
        for line in root.findall(".//*[@id='edu.mit.parsing.parse']/desc"):
            off_parsing = int(line.get("off"))
            len_parsing = int(line.get("len"))
            sentspan_to_sexp[(off_parsing,len_parsing)] = line.text
        
        for sentence_id, sentence_instance in self.sentences.items():
            sexp = sentspan_to_sexp[(sentence_instance.offset, sentence_instance.len)]
            sentence_instance.sexpression = sexp
        
        # prepare functions which is needed for processing SRL annotation info
        def identify_subseq(original_seq: List[int], sub_seq: List[int]):
            return "".join([str(i) for i in sub_seq]) in "".join([str(i) for i in original_seq])
        
        def get_sentid_from_textspan(sentspan2sentid, off, length):
            for span, sent_id in sentspan2sentid.items():
                if identify_subseq(range(span[0], span[0]+span[1]), range(off, off+length)):
                    return sent_id
            raise ValueError
            return None
        
        #prosessing Predicate Argument Structure (PAS) infomation
        sent_id_to_SRL_list = defaultdict(list)
        srl_id_to_srlinfo = {}

        for line in root.findall(".//*[@id='edu.mit.semantics.semroles']/desc"):
            id_SRL = int(line.get("id"))
            off_SRL = int(line.get("off"))
            len_SRL = int(line.get("len"))

            sent_id = get_sentid_from_textspan(self.sentspan2sentid, off_SRL, len_SRL)
            self.sentid2srls[sent_id].append(id_SRL)
            self.srls[id_SRL] = ProppLearnerSRL(line.text, off_SRL, len_SRL, self.sentences[sent_id])
            
        for srl_id, srl in self.srls.items():
            srl.sentence_id = self.tokenid2sentid[srl.pred_token_id]
            
        for sentence_id, sentence_instance in self.sentences.items():
            #sexp = sentspan_to_sexp[(sentence_instance.offset, sentence_instance.len)]
            for srl_id in self.sentid2srls[sentence_id]:
                sentence_instance.srllist.append(self.srls[srl_id])
                
        # prosessing Event infomation
        for line in root.findall(".//*[@id='edu.mit.semantics.rep.event']/desc"):
            id_event = int(line.get("id"))
            off_event = int(line.get("off"))
            len_event = int(line.get("len"))
            
            event_type, head, span, pos, *_ = line.text.split("|")
            
            if "," in span:
                self.event_tokens_list.append(list(flatten([[int(i) for i in sub_span.split("~")] for sub_span in span.split(",")])))
            else:
                self.event_tokens_list.append([int(i) for i in span.split("~")])
            
        for sentence_id, sentence_instance in self.sentences.items():
            for event_tokens_list in self.event_tokens_list:
                if all([token_id in sentence_instance.tokens for token_id in event_tokens_list]):
                    sentence_instance.events.append(event_tokens_list)
                #elif any([token_id in sentence_instance.tokens for token_id in event_tokens_list]) and [token_id in sentence_instance.tokens for token_id in event_tokens_list].count(True) != len(event_tokens_list):
                    #print("Attention!")
                else:
                    pass
        #print(len(root.findall(".//*[@id='edu.mit.semantics.rep.event']/desc")))
        
        #prosessing proppian functions annotation infomation
        def get_sent_num(token_id, sent_identifier):
            sent_num = -1
            for key, value in sent_identifier.items():
                if int(token_id) in value:
                    sent_num = key
            return sent_num
        
        #proppian_functions = {}
        for line in root.findall(".//*[@id='edu.mit.semantics.rep.function']/desc"):
            description = line.text.split("|")

            if description[0] == "alpha":
                #print("contains alpha")
                pass
            else:
                statement_list = [re.split("[:,~]", item) for item in description[1:]]
                self.functions[description[0]] = [[int(i) for i in statement[1:]] if statement[0] == "ACTUAL" else [] for statement in statement_list]

        for sentence_id, sentence_instance in self.sentences.items():
            for function_type, token_id_list_list in self.functions.items():
                for token_id_list in token_id_list_list:
                    if token_id_list and all([token_id in sentence_instance.tokens for token_id in token_id_list]):
                        sentence_instance.functions[function_type].append(token_id_list)
                    else:
                        pass

if __name__ == '__main__':
    pass