"""Question-disjoint data preparation and final-answer labels."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from .io import read_json,write_json,require_empty,sha256

NUMBER=re.compile(r'[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?')


def number(value):
    text=str(value).strip()
    return Decimal(text.replace(',','')) if NUMBER.fullmatch(text) else None


def family(text):
    text=unicodedata.normalize('NFKC',text).lower()
    return re.sub(r'\W+',' ',re.sub(r'\d+(?:[.,]\d+)*',' NUMBER ',text)).strip()


def reference(problem,solution,saved):
    boxes=[]
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{',solution):
        start=match.end();depth=1
        for i in range(start,len(solution)):
            depth+=(solution[i]=='{')-(solution[i]=='}')
            if depth==0:boxes.append(solution[start:i]);break
        else:return None
    result=number(boxes[0]) if len(boxes)==1 else None
    if result is None or number(saved)!=result:return None
    if re.search(r'(?:find|enter|give|list|determine)\s+all\s+(?:the\s+)?(?:possible\s+)?(?:values|solutions|roots)|separated\s+by\s+commas',problem,re.I):return None
    return str(result)


def load_questions(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids=set();families=set()
    for row in rows:
        if row.get('split') not in ('train','val','test') or not row.get('id') or not row.get('question'):
            raise ValueError('Each question needs id, question, reference and split=train|val|test')
        if number(row.get('reference')) is None:raise ValueError('Reference must be a single decimal number')
        f=family(row['question'])
        if row['id'] in ids or f in families:raise ValueError('Duplicate ID or number-template family')
        ids.add(row['id']);families.add(f)
    if not rows or {r['split'] for r in rows}!={'train','val','test'}:raise ValueError('All three splits are required')
    return rows


def prepare_numina(out,counts,revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset='AI-MO/NuminaMath-1.5'
    revision=HfApi().dataset_info(dataset, revision=revision).sha
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Dataset resolver did not return an immutable commit SHA')
    rows=load_dataset(dataset,split='train',revision=revision)
    selected={};excluded=Counter()
    for i,row in enumerate(rows):
        if (row.get('source')!='cn_k12' or row.get('synthetic') is not False or
            row.get('question_type')!='math-word-problem' or row.get('problem_is_valid')!='Yes' or row.get('solution_is_valid')!='Yes'):
            excluded['metadata']+=1;continue
        question=row['problem'].strip()
        if re.search(r'!\[|\\includegraphics|https?://|\b(?:diagram|figure|shown below|prove that)\b',question,re.I):
            excluded['non_text_or_proof']+=1;continue
        target=reference(question,row['solution'],row.get('answer',''))
        if target is None:excluded['reference']+=1;continue
        key=family(question)
        if key in selected:excluded['duplicate_family']+=1;continue
        selected[key]=dict(id=f'numina-cn-k12/{i}',question=question,reference=target)
    ordered=sorted(selected.items(),key=lambda pair:hashlib.sha256(('slice-v1:'+pair[0]).encode()).hexdigest())
    if any(v<1 for v in counts.values()) or len(ordered)<sum(counts.values()):raise ValueError('Not enough eligible families for requested positive split sizes')
    out=require_empty(out);cursor=0;questions=[]
    for split,count in counts.items():
        questions.extend(dict(row,split=split) for _,row in ordered[cursor:cursor+count]);cursor+=count
    (out/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
    write_json(out/'selection.json',dict(dataset=dataset,revision=revision,counts=counts,eligible=len(ordered),exclusions=dict(excluded),questions_sha256=sha256(out/'questions.jsonl'),
        scope='One number-template family per question; not semantic or pretraining decontamination. New selection, not the original reported benchmark.'))


def grade(text,target):
    from math_verify import parse,verify,LatexExtractionConfig,ExprExtractionConfig
    from math_verify.errors import TimeoutException
    from latex2sympy2_extended import NormalizationConfig
    from sympy import Rational,SympifyError
    if number(target) is None:raise ValueError('Expected a decimal reference')
    result=dict(score=None,status='parse_failure',extracted=None)
    if len(text)>32000:return dict(result,status='input_too_long')
    # A scalar-target task must not silently accept one of several final boxes.
    boxes=[]
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{',text):
        start=match.end();depth=1
        for i in range(start,len(text)):
            depth+=(text[i]=='{')-(text[i]=='}')
            if depth==0:boxes.append(text[start:i].strip());break
        else:return dict(result,status='malformed_box')
    if len(set(boxes))>1:return dict(result,status='ambiguous_final')
    # Restrict this guard to explicitly announced final answers, not intermediate
    # arithmetic in the reasoning. A list of alternatives is not a scalar answer.
    announcements = list(re.finditer(r'\b(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n]*)', text, re.I))
    if announcements:
        tail = announcements[-1].group(1).strip()
        scalar = r'[+-]?(?:[0-9]+(?:\.[0-9]+)?)'
        alternatives = rf'{scalar}\s*(?:,\s+|\bor\b\s*|\band\b\s*){scalar}(?![0-9])'
        if re.search(alternatives, tail, re.I):
            return dict(result, status='ambiguous_final')
        # Do not accept an old boxed number when a later explicit scalar answer
        # contradicts it. More general symbolic ambiguity remains an audit issue.
        final_number = NUMBER.match(tail.lstrip('$ ('))
        boxed_number = number(boxes[-1]) if boxes else None
        if boxed_number is not None and final_number and number(final_number.group()) != boxed_number:
            return dict(result, status='ambiguous_final')
    extraction=[LatexExtractionConfig(boxed_match_priority=0,try_extract_without_anchor=True,
        normalization_config=NormalizationConfig(basic_latex=True,units=True,malformed_operators=False,nits=False,boxed='all',equations=False)),
        ExprExtractionConfig(try_extract_without_anchor=True)]
    try:
        prediction=parse(text,extraction_config=extraction,fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=3,raise_on_error=True)
    except TimeoutException:return dict(result,status='parse_timeout')
    except SympifyError:return result
    except (MemoryError,OSError,ImportError):raise
    except Exception as exc:return dict(result,status='parse_error',error=type(exc).__name__)
    if not prediction:return result
    if len(prediction)!=1 or isinstance(prediction[0],str):return dict(result,status='unsupported_parse')
    if not getattr(prediction[0],'is_number',False):return dict(result,status='unsupported_parse')
    result['extracted']=str(prediction[0])
    try:
        correct=verify(Rational(str(number(target))),prediction[0],float_rounding=6,numeric_precision=15,strict=True,timeout_seconds=3,raise_on_error=True)
    except TimeoutException:return dict(result,status='verify_timeout')
    except (MemoryError,OSError,ImportError):raise
    except Exception as exc:return dict(result,status='verify_error',error=type(exc).__name__)
    return dict(result,score=int(correct),status='correct' if correct else 'incorrect')
