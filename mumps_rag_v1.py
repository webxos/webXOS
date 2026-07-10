#!/usr/bin/env python3
#!/usr/bin/env python3
import re
import math
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# Persistent storage file
STORAGE_FILE = "mumps_globals.json"

class MUMPSGlobals:
    def __init__(self):
        self.globals = defaultdict(dict)
        self.load()
    
    def load(self):
        
        if Path(STORAGE_FILE).exists():
            try:
                with open(STORAGE_FILE, 'r') as f:
                    data = json.load(f)
                    self.globals = defaultdict(dict, {k: dict(v) for k, v in data.items()})
                print(f"[Loaded {sum(len(v) for v in self.globals.values())} entries]")
            except Exception as e:
                print(f"[Load error: {e}]")
        else:
            self._preload_defaults()
    
    def save(self):
        
        try:
            with open(STORAGE_FILE, 'w') as f:
                json.dump(self.globals, f, indent=2)
        except Exception as e:
            print(f"[Save error: {e}]")
    
    def _preload_defaults(self):
        
        docs = [
            (1, "SOAP framework: Subjective, Objective, Assessment, Plan. Used for progress notes and communication."),
            (2, "ABCDE framework: Airway, Breathing, Circulation, Disability, Exposure. Used for emergency assessment and trauma."),
            (3, "Chest pain assessment: characterize pain, associated symptoms, risk factors, vital signs, ECG, troponin."),
            (4, "Emergency billing: 99282 to 99285 for ED visits, level based on history, exam, and medical decision making."),
            (5, "Diabetes follow-up visit: check A1C, medications, hypoglycemia, complications screening, foot exam, vaccinations."),
        ]
        for doc_id, text in docs:
            self.globals['DOC'][str(doc_id)] = text
        self.save()
    
    def set_value(self, root, subscripts, value):
        
        current = self.globals[root]
        for sub in subscripts[:-1]:
            if sub not in current:
                current[sub] = {}
            current = current[sub]
            if not isinstance(current, dict):
                raise ValueError(f"Path conflict: {root}({','.join(subscripts)})")
        current[subscripts[-1]] = value
        self.save()  
    
    def get_value(self, root, subscripts):
        
        current = self.globals.get(root, {})
        for sub in subscripts:
            if not isinstance(current, dict):
                return None
            current = current.get(sub)
            if current is None:
                return None
        return current
    
    def list_tree(self, root, subscripts=()):
        
        current = self.globals.get(root, {})
        for sub in subscripts:
            if isinstance(current, dict):
                current = current.get(sub, {})
            else:
                return []
        
        results = []
        if isinstance(current, dict):
            for k, v in current.items():
                full_key = subscripts + (k,)
                if isinstance(v, dict):
                    results.extend(self.list_tree(root, full_key))
                else:
                    results.append((root, full_key, v))
        return results
    
    def get_all_docs(self):
        
        return self.list_tree('DOC')

db = MUMPSGlobals()

# TF-IDF with Cosine Similarity
def tokenize(text):
    
    return re.findall(r'\w+', str(text).lower())

def compute_tfidf_cosine(q_tokens, doc_list):
    
    if not doc_list:
        return []
    
    
    doc_tokens_list = [tokenize(text) for _, _, text in doc_list]
    
    
    vocab = set(q_tokens)
    if not vocab:
        return [0.0] * len(doc_list)
    
    
    doc_freq = defaultdict(int)
    for doc_tokens in doc_tokens_list:
        unique_in_doc = set(doc_tokens)
        for token in vocab:
            if token in unique_in_doc:
                doc_freq[token] += 1
    
    
    num_docs = len(doc_list)
    idf = {}
    for token in vocab:
        df = doc_freq.get(token, 0)
        
        idf[token] = math.log((num_docs + 1) / (df + 1)) if df > 0 else 0.0
    
    
    q_counter = Counter(q_tokens)
    q_tfidf = {}
    for token in vocab:
        tf = q_counter.get(token, 0)
        q_tfidf[token] = tf * idf[token]
    
    
    q_magnitude = math.sqrt(sum(v ** 2 for v in q_tfidf.values()))
    if q_magnitude == 0:
        return [0.0] * len(doc_list)
    
    
    scores = []
    for doc_tokens in doc_tokens_list:
        doc_counter = Counter(doc_tokens)
        
        
        doc_tfidf = {}
        for token in vocab:
            tf = doc_counter.get(token, 0)
            doc_tfidf[token] = tf * idf[token]
        
        
        doc_magnitude = math.sqrt(sum(v ** 2 for v in doc_tfidf.values()))
        if doc_magnitude == 0:
            scores.append(0.0)
            continue
        
        
        dot_product = sum(q_tfidf[t] * doc_tfidf.get(t, 0) for t in q_tfidf)
        cosine_sim = dot_product / (q_magnitude * doc_magnitude)
        scores.append(cosine_sim)
    
    return scores

def rag_query(q, n=5):
    
    docs = db.get_all_docs()
    
    if not docs:
        return []
    
    q_tokens = tokenize(q)
    if not q_tokens:
        return []
    
    
    scores = compute_tfidf_cosine(q_tokens, docs)
    
    
    results = []
    for (root, subs, text), score in zip(docs, scores):
        if score > 0:
            key_str = f"^{root}({','.join(subs)})"
            results.append((key_str, text, score))
    
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:n]

def parse_key(key_str):
    
    key_str = key_str.strip()
    if not key_str.startswith('^'):
        raise ValueError("Key must start with ^")
    
    m = re.match(r'^\^([A-Za-z0-9]+)\((.*)\)$', key_str)
    if m:
        root, subs_str = m.groups()
        subs = [s.strip().strip('"') for s in subs_str.split(',')]
        return root, subs
    else:
        m = re.match(r'^\^([A-Za-z0-9]+)$', key_str)
        if m:
            return m.group(1), []
        raise ValueError(f"Invalid key format: {key_str}")


print("*** MUMPS RAG TERMINAL (by webXOS 2025) ***")
print("Type HELP for commands\n")

while True:
    try:
        line = input("USER> ").strip()
        if not line:
            continue
        
        parts = re.split(r'\s+', line, maxsplit=1)
        cmd = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""
        
        if cmd == "HELP":
            print("Commands:")
            print("  SET ^root(sub)=value           Set global variable [SAVES]")
            print("  GET ^root(sub)                 Get global variable")
            print("  LIST ^root(sub)                List subtree")
            print("  DOC SHOW                       Show all documents")
            print("  DOC ADD ^DOC(id)=text          Add new document [SAVES]")
            print("  RAG QUERY text                 Retrieve relevant docs (TF-IDF cosine)")
            print("  EXIT                           Quit")
        
        elif cmd == "SET":
            m = re.match(r'^(\^[^^=]+)=(.*)$', rest)
            if m:
                key_str, val = m.groups()
                root, subs = parse_key(key_str)
                db.set_value(root, subs, val.strip())  # Saves internally
                print(f"SET {key_str}")
            else:
                print("Syntax: SET ^root(sub1,sub2,...)=value")
        
        elif cmd == "GET":
            if not rest:
                print("Syntax: GET ^root(sub1,sub2,...)")
                continue
            root, subs = parse_key(rest)
            val = db.get_value(root, subs)
            key_str = f"^{root}({','.join(subs)})" if subs else f"^{root}"
            print(f"{key_str} = {val if val is not None else '<undefined>'}")
        
        elif cmd == "LIST":
            if not rest:
                print("Syntax: LIST ^root(sub1,sub2,...)")
                continue
            root, subs = parse_key(rest)
            items = db.list_tree(root, tuple(subs))
            if items:
                key_str = f"^{root}({','.join(subs)})" if subs else f"^{root}"
                print(f"Tree {key_str}:")
                for r, subscripts, val in items:
                    full_key = f"^{r}({','.join(subscripts)})"
                    short_val = val[:80] + ("..." if len(val) > 80 else "")
                    print(f"  {full_key} = {short_val}")
            else:
                print("Empty")
        
        elif cmd == "DOC":
            parts_doc = rest.split(None, 1)
            sub = parts_doc[0].upper() if parts_doc else ""
            
            if sub == "SHOW":
                items = db.get_all_docs()
                if items:
                    print("Documents:")
                    for r, subs, text in items:
                        short = text[:100] + ("..." if len(text) > 100 else "")
                        print(f"  ^{r}({','.join(subs)}) = {short}")
                else:
                    print("No documents")
            
            elif sub == "ADD":
                rest_add = parts_doc[1] if len(parts_doc) > 1 else ""
                m = re.match(r'^(\^DOC\([^)]+\))=(.*)$', rest_add)
                if m:
                    key_str, val = m.groups()
                    root, subs = parse_key(key_str)
                    db.set_value(root, subs, val.strip())  
                    print("DOC ADDED")
                else:
                    print("Syntax: DOC ADD ^DOC(id)=text")
            else:
                print("Subcommands: DOC SHOW, DOC ADD")
        
        elif cmd == "RAG":
            if not rest.upper().startswith("QUERY "):
                print("Syntax: RAG QUERY text")
                continue
            q = rest[6:].strip()
            print(f"RAG: {q}")
            results = rag_query(q)
            if results:
                for i, (key, text, score) in enumerate(results, 1):
                    short = text[:100] + ("..." if len(text) > 100 else "")
                    print(f"  #{i} [{key}] cosine={score:.4f}")
                    print(f"      {short}")
            else:
                print("No matches")
        
        elif cmd in ("EXIT", "QUIT"):
            print("Bye")
            break
        
        else:
            print("Unknown command. Type HELP.")
    
    except KeyboardInterrupt:
        print("\nBye")
        break
    except Exception as e:
        print(f"ERR: {e}")