"""Oracle-program arithmetic and retrieval-grounding audit.

IMPORTANT: FinQA gold programs are used ONLY in this evaluator. This is not a
natural-language end-to-end QA score and does not measure independent generation.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import json
from financial_research.arithmetic import InvalidCalculation, execute, all_literals_grounded, reference_agrees
from financial_research.corpus import ROOT, load_finqa


def evaluate():
    rows=json.loads((ROOT/'data/finqa-dev.json').read_text()); _,queries=load_finqa()
    lookup={x['id']:x for x in queries if x['split']=='test'}
    by_chunk={}
    chunks,_=load_finqa()
    by_chunk={x.id:x.text for x in chunks}
    predict=json.loads((ROOT/'reports/predictions.json').read_text())
    retrieved={(x['query_id'],x['scope'],x['method']):x['predicted']
               for x in predict if x['split']=='test'}
    records=[]
    for row in rows:
        if row['id'] not in lookup:continue
        q=lookup[row['id']]
        program=row['qa']['program']; result={'query_id':q['id'],'source_page':q['document'],'reference_answer':q['reference_answer']}
        try:
            value,trace=execute(program)
            result.update(supported=True,program_result=value,trace=trace,
                          agrees_with_exe_ans=reference_agrees(value,row['qa']['exe_ans']),
                          agrees_with_answer=reference_agrees(value,row['qa']['answer']))
            result['oracle_gold_evidence_grounded']=all_literals_grounded(program,[by_chunk[g] for g in q['gold']])
            for scope in ('collection','page'):
                for method in ('bm25','reranked'):
                    ids=retrieved.get((q['id'],scope,method),[])
                    result[f'{scope}_{method}_operand_covered']=all_literals_grounded(program,[by_chunk[i] for i in ids])
        except (InvalidCalculation,ValueError) as exc:
            result.update(supported=False,reason=str(exc))
        records.append(result)
    def count(key):return sum(x.get(key) is True for x in records)
    supported=count('supported');n=len(records)
    summary={'benchmark':'FinQA custom report-disjoint test repartition','test_questions':n,
         'supported_programs':supported,'unsupported_programs':n-supported,
         'program_agrees_with_exe_ans':count('agrees_with_exe_ans'),
         'program_agrees_with_reported_answer':count('agrees_with_answer'),
         'oracle_gold_evidence_operand_coverage':count('oracle_gold_evidence_grounded'),
         'collection_bm25_operand_coverage':count('collection_bm25_operand_covered'),
         'collection_reranked_operand_coverage':count('collection_reranked_operand_covered'),
         'page_bm25_operand_coverage':count('page_bm25_operand_covered'),
         'page_reranked_operand_coverage':count('page_reranked_operand_covered'),
         'denominator':'All 162 held-out questions, including unsupported DSL forms.',
         'critical_limit':'All program-execution statistics use GOLD FinQA programs (oracle inputs). None measure question-to-program generation, retrieved-cell selection, end-to-end QA accuracy or free-text faithfulness.',
         'annotation_limit':'Program results, exe_ans and written answers can disagree. Numeric comparisons tolerate common percent scales and reasonable rounding; these are diagnostics, not adjudicated labels.'}
    report=ROOT/'reports'; (report/'arithmetic-evaluation.json').write_text(json.dumps({'summary':summary,'examples':records},indent=2))
    lines=['# Numerical reasoning: oracle-program audit','','This is a **gold-program diagnostic**, *not* end-to-end QA. The evaluator receives the answer program given by FinQA; the live application does not read these programs.','',
        '| Check | Count | Denominator |','|---|---:|---:|']
    for k in ('supported_programs','program_agrees_with_exe_ans','program_agrees_with_reported_answer',
              'oracle_gold_evidence_operand_coverage','collection_bm25_operand_coverage','collection_reranked_operand_coverage','page_bm25_operand_coverage'):
        lines.append(f"| {k.replace('_',' ')} | {summary[k]} | {n} |")
    lines += ['','## Interpretation','','- Arithmetic is executed by a bounded, pure whitelist DSL; no Python `eval`, no arbitrary code, no network.','- Operand coverage checks whether each literal number occurs in at least one cited passage; it does **not** confirm cell/column attribution, sign, scale or unit consistency.','- FinQA has internally inconsistent example annotations; report disagreements and retain examples. Do not silently overwrite the benchmark labels.','- A production-worthy numerical QA system must select operands and operations from the *question* without oracle programs; this audit cannot establish that capability.','- Real answer-quality and faithfulness evaluation for model-generated answers remains unverified without real generated model outputs.']
    (report/'ARITHMETIC.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':evaluate()
