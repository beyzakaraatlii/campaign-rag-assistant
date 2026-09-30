"""Opt-in real-model smoke test. Run: python -m tests.smoke_local [--with-llm]."""
import argparse
import json
import tempfile
from pathlib import Path

from config import PROJECT_ROOT
from database import rebuild_database
from llm import answer_from_results
from retrieval import retrieve
from uploads import save_uploaded_pdf
from tests.helpers import pdf_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-llm', action='store_true')
    args = parser.parse_args()
    root = PROJECT_ROOT / '.test-artifacts'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        folder = Path(directory) / 'documents'
        db = Path(directory) / 'index.db'
        save_uploaded_pdf('coffee.pdf', pdf_bytes('Coffee is prepared with hot water and roasted beans.'), folder)
        rebuild_database(folder, db)
        save_uploaded_pdf('new_campaign.pdf', pdf_bytes(
            'The ZEBRA advertising campaign has the unique campaign code ORBIT-739. '
            'The ZEBRA campaign launch date is October 15.'), folder)
        count = rebuild_database(folder, db)
        question = 'What is the unique code of the ZEBRA advertising campaign?'
        results = retrieve(question, db)
        assert results and results[0]['source'] == 'new_campaign.pdf', results
        assert 'ORBIT-739' in results[0]['text']
        turkish_results = retrieve('ZEBRA reklam kampanyasının özel kodu nedir?', db)
        assert turkish_results and turkish_results[0]['source'] == 'new_campaign.pdf', turkish_results
        report = {'real_embeddings': True, 'chunks': count,
                  'top_source': results[0]['source'], 'top_score': results[0]['score'],
                  'turkish_retrieval': True,
                  'real_phi4': False}
        try:
            if args.with_llm:
                print('Loading Phi-4-mini into the project-local cache...', flush=True)
                answer, sources = answer_from_results(None, question, results)
                report.update(answer=answer, sources=sources)
                assert 'ORBIT-739' in answer, answer
                report['real_phi4'] = True
        except Exception as exc:
            report['error'] = str(exc).splitlines()[0]
            raise
        finally:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            (root / 'smoke-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
