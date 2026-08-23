#!/usr/bin/env python3
import re
import math
from collections import Counter

db = {}

def tokenize(text):
    return re.findall(r'\w+', str(text).lower())

def score_doc(q_tokens, d_tokens):
    if not d_tokens: return 0.0
    qf = Counter(q_tokens)
    df = Counter(d_tokens)
    score = sum(math.sqrt(qf[t] * df.get(t, 0)) for t in qf)
    return score / math.sqrt(len(d_tokens))

def flat_key(k):
    k = k.lstrip('^')
    m = re.match(r'^([A-Za-z0-9]+)(?:\((.*)\))?$', k)
    if not m: return k
    base, subs = m.groups()
    if subs:
        subs = [s.strip().strip('"') for s in subs.split(',')]
        return '|'.join([base] + subs)
    return base

def unflat_key(f):
    p = f.split('|')
    if len(p) == 1: return '^' + p[0]
    return '^' + p[0] + '(' + ','.join(f'"{x}"' if ' ' in x else x for x in p[1:]) + ')'

def preload():
    if any(k.startswith('DOC') for k in db):
        return
    docs = [
        ("^DOC(1)", "SOAP framework: Subjective, Objective, Assessment, Plan. Used for progress notes and communication."),
        ("^DOC(2)", "ABCDE framework: Airway, Breathing, Circulation, Disability, Exposure. Used for emergency assessment and trauma."),
        ("^DOC(3)", "Chest pain assessment: characterize pain, associated symptoms, risk factors, vital signs, ECG, troponin."),
        ("^DOC(4)", "Emergency billing: 99282 to 99285 for ED visits, level based on history, exam, and medical decision making."),
        ("^DOC(5)", "Diabetes follow-up visit: check A1C, medications, hypoglycemia, complications screening, foot exam, vaccinations."),
    ]
    for k, v in docs:
        db[flat_key(k)] = v

def rag(q, n=5):
    docs = [(k, v) for k, v in db.items() if k.startswith('DOC')]
    qt = tokenize(q)
    res = []
    for f, t in docs:
        s = score_doc(qt, tokenize(t))
        if s > 0: res.append((unflat_key(f), t, s))
    res.sort(key=lambda x: x[2], reverse=True)
    return res[:n]

preload()
print("*** MUMPS RAG TERMINAL (by webXOS 2025) ***\nType HELP")

while True:
    try:
        line = input("USER> ").strip()
        if not line: continue
        parts = re.split(r'\s+', line)
        cmd = parts[0].upper()

        if cmd == "HELP":
            print("SET ^G(k)=v  GET ^G(k)  LIST root  DOC ADD ^DOC(id)=text  DOC SHOW  RAG QUERY text  EXIT")

        elif cmd == "SET":
            rest = ' '.join(parts[1:])
            m = re.match(r'^(\^[^^]+)=(.*)$', rest)
            if m:
                key, val = m.groups()
                db[flat_key(key)] = val.strip()
                print(f"SET {key}")
            else:
                print("Bad SET syntax")

        elif cmd == "GET":
            if len(parts) < 2:
                print("Need key")
                continue
            k = parts[1]
            v = db.get(flat_key(k))
            print(f"{k} = {v if v is not None else '<undef>'}")

        elif cmd == "LIST":
            if len(parts) < 2:
                print("Need root")
                continue
            r = parts[1].upper()
            items = [(k, v) for k, v in db.items() if k == r or k.startswith(r + '|')]
            if items:
                print(f"ROOT {r}:")
                for f, v in items:
                    print(f"  {unflat_key(f)} = {v}")
            else:
                print("Empty")

        elif cmd == "DOC":
            if len(parts) < 2:
                print("DOC SHOW or DOC ADD")
                continue
            sub = parts[1].upper()
            if sub == "SHOW":
                items = [(k, v) for k, v in db.items() if k.startswith('DOC')]
                if not items:
                    print("No DOCs")
                for f, v in items:
                    print(f"  {unflat_key(f)} = {v}")
            elif sub == "ADD":
                rest = ' '.join(parts[2:])
                m = re.match(r'^(\^DOC\([^)]+\))=(.*)$', rest, re.I)
                if m:
                    key, val = m.groups()
                    db[flat_key(key)] = val.strip()
                    print("DOC ADDED")
                else:
                    print("Bad ADD syntax")
            else:
                print("Unknown DOC subcommand")

        elif cmd == "RAG":
            if len(parts) < 3 or parts[1].upper() != "QUERY":
                print("RAG QUERY text")
                continue
            q = ' '.join(parts[2:])
            print(f"RAG: {q}")
            results = rag(q)
            if not results:
                print("No matches")
            for i, (k, t, s) in enumerate(results, 1):
                short = t[:157] + ("..." if len(t)>160 else "")
                print(f"  #{i} [{k}] {s:.3f}\n      {short}")

        elif cmd in ("EXIT", "QUIT"):
            break

        else:
            print("Unknown command")

    except KeyboardInterrupt:
        print("\nBye")
        break
    except Exception as e:
        print(f"ERR: {e}")
