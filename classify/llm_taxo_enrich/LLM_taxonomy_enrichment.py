
import argparse 
import tqdm
from openai import OpenAI
import re
import os
import string

class Node:
    def __init__(self, node_id, name):
        self.name = name
        self.node_id = node_id
        self.parents = []
        self.childs = []
        self.similarity_score = 0
        self.path_score = 0

    def addChild(self, child):
        if child not in self.childs:
            self.childs.append(child)

    def addParent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)

    def findChild(self, node_id):
        if node_id == self.node_id:
            return self
        if len(self.childs) == 0:
            return None
        for child in self.childs:
            ans = child.findChild(node_id)
            if ans != None:
                return ans
        return None
    
def parse_key_terms(original_string):
    """
    Parses a string of comma-separated key terms into a list.

    Args:
    terms_str (str): A string of comma-separated key terms related to hair care.

    Returns:
    list: A list containing individual key terms.
    """
    # Strip whitespace and split the string by commas
    terms_str = original_string.split("terms: ")[1].replace("\n","")
    terms_str = terms_str.split('.')[0]
    terms_str = re.sub(r'\d+\.', ',', terms_str)
    print(terms_str)
    key_terms_list = [term.strip().lower().translate(str.maketrans('', '', string.punctuation)) for term in terms_str.split(',')]
    key_terms_list = [i.strip() for i in key_terms_list if i !=""]
    print(key_terms_list)
    return key_terms_list

def generate_enrichment(class_name,siblings,parent_class,claim: str):

    out = None
    client = OpenAI()

    parent_prompt = f" and is the subaspect of node '{', '.join(parent_class)}'" if parent_class!=[] else ""
    
    message = [
            {"role": "user", "content": f"'{class_name}' is an aspect when analyzing the claim: {claim}{parent_prompt}. Please generate 10 additional key terms about the '{class_name}' class that is relevant to '{class_name}' but irrelevant to {siblings}. Please split the additional key terms using commas and output in the form: Key terms:\n(key terms)."}
            ]
    
    results_raw = ""
    tries = 0
    while out == None or (':' not in results_raw):
        try:
            results = client.chat.completions.create(
                model='gpt-4o',
                messages=message,
                temperature=0.6,
                top_p=0.9,
            )
            results_raw = results.choices[0].message.content.strip('\n')
            key_term = parse_key_terms(results_raw)
            key_term = [i.replace(' ', '_') for i in key_term]
            out = class_name.replace(' ', '_')+":"+",".join(key_term)
            print('Has Keywords')
            print(out)
        except:
            tries += 1
            if tries > 5:
                out = class_name.replace(' ', '_') + ":"
                print('Without Keywords')
                print(out)
                return out
            print('try again')
    
    return out

#create taxonomy graph
def createGraph(args):
    root = None

    id2label = {}
    label2id = {}
    with open(args.label_path) as f:
        for line in f:
            label_id, label_name = line.strip().split('\t')
            id2label[label_id] = label_name
            label2id[label_name] = label_id

    # construct graph from file
    with open(args.label_hierarchy_path) as f:
        ## for each line in the file
        root = Node(-1, 'ROOT')
        for line in f:
            parent_id, child_id = line.strip().split('\t')
            parent = id2label[parent_id]
            child = id2label[child_id]
            parent_node = root.findChild(parent_id)
            if parent_node is None:
                parent_node = Node(parent_id, parent)
                root.addChild(parent_node)
                parent_node.addParent(root)
            child_node = root.findChild(child_id)
            if child_node is None:
                child_node = Node(child_id, child)
            parent_node.addChild(child_node)
            child_node.addParent(parent_node)
    
    return root, id2label, label2id

def get_sib(root:Node,idx):
    cur_node = root.findChild(idx)
    par_node = cur_node.parents
    sib = []
    for j in par_node:
        sib.extend([i for i in j.childs if i != cur_node])
    return [i.name.replace("_"," ") for i in sib]

def get_par(root:Node,idx):
    cur_node = root.findChild(idx)
    parent_list = []
    # assert len(cur_node.parents)==1
    for par_node in cur_node.parents:
        if par_node != root:
            parent_list.append(par_node.name)
            for par_par_node in par_node.parents:
                if par_par_node != root:
                    parent_list.append(par_par_node.name)
    return [i.replace("_"," ") for i in parent_list]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--label_path', type=str, help='path to label.txt, lines: id tlabel')
    parser.add_argument('--label_hierarchy_path', type=str, help='path to label_hierarchy.txt, lines: parent_id\tchild_id')
    parser.add_argument('--claim_path', type=str, help='path to claim.txt, lines: claim')
    parser.add_argument('--output_path', type=str, help='path to data directory')
    args = parser.parse_args()

    # read taxonomy graph
    root, id2label, label2id = createGraph(args)
    num_class = len(id2label)

    # generate LLM-based taxonomy enrichment
    enrichment_list = []
    
    # load the claim from the file
    with open(args.claim_path) as f:
        claim = f.readlines()[0].strip()
    
    with open(args.output_path, 'w') as fout1:
        for i in tqdm.tqdm(range(num_class)):
            siblings = get_sib(root,str(i))
            class_name = root.findChild(str(i)).name.replace("_"," ")
            parent_class = get_par(root,str(i))
            enrichment = generate_enrichment(class_name,siblings,parent_class, claim)
            fout1.write(enrichment+'\n')
            enrichment_list.append(enrichment)
    