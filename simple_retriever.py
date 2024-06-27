#%%
import logging
import sys
from typing import List, Tuple
from pyserini.search import SimpleSearcher
from wepyserini.retriever_utils import load_passages, validate, save_results
import os
import pickle
import csv

os.environ["PYSERINI_CACHE"] = "path-to-cache/bm25_cache"

def setup_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 输出到文件的 handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 输出到控制台的 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def save_data_with_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

class SparseRetriever:
    def __init__(self, index_name, log_path,  num_threads=1):
        self.searcher = SimpleSearcher.from_prebuilt_index(index_name)
        self.num_threads = num_threads
        self.dedup = False
        self.logger = setup_logger(log_path)
        
    def get_top_docs(
        self, questions: List[str], top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:

        qids = [str(x) for x in range(len(questions))]
        if self.dedup:
            dedup_q = {}
            dedup_map = {}
            for qid, question in zip(qids, questions):
                if question not in dedup_q:
                    dedup_q[question] = qid
                else:
                    dedup_map[qid] = dedup_q[question]
            dedup_questions = []
            dedup_qids = []
            for question in dedup_q:
                qid = dedup_q[question]
                dedup_questions.append(question)
                dedup_qids.append(qid)
            self.logger.info(f"Deduplicate from {len(qids)} to {len(dedup_qids)}.")
            hits = self.searcher.batch_search(queries=dedup_questions, qids=dedup_qids, k=top_docs, threads=self.num_threads)
            for qid in dedup_map:
                hits[qid] = hits[dedup_map[qid]]
        else:
            hits = self.searcher.batch_search(queries=questions, qids=qids, k=top_docs, threads=self.num_threads)
        results = []
        for qid in qids:
            example_hits = hits[qid]
            example_top_docs = [hit.docid for hit in example_hits]
            example_scores = [hit.score for hit in example_hits]
            results.append((example_top_docs, example_scores))
        return results



# %%
if __name__ == "__main__":
    
    log_path = 'log_path/log.txt'
    top_docs_pkl_path = 'path-to-output/top_docs_pkl'
    input_file_path = 'path-to-input/nq-test.csv'
    
    logger = setup_logger(log_path)
    logger.info("Validation Start!")

    with open(input_file_path,'r') as file:
        query_data = csv.reader(file, delimiter='\t')
        questions, question_answers = zip(*[(item[0], eval(item[1])) for item in query_data])
        questions = questions[0:1]
        question_answers = question_answers[0:1]
    logger.info(f"""questions: {questions[0]}""")

    index_name = "wikipedia-dpr"  
    retriever = SparseRetriever(index_name,log_path)

    top_docs_list = retriever.get_top_docs(questions)
    logger.info(top_docs_list)

    os.makedirs(top_docs_pkl_path, exist_ok=True)
    save_data_with_pickle(top_docs_list, os.path.join(top_docs_pkl_path,'top_docs.pkl'))
    logger.info("First Retrieval and Updated Retrieval Completed!")
