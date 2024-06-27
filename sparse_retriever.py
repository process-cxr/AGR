"""
 Command line tool to get dense results and validate them
"""

import argparse
import glob
import logging
import os
import sys
import time
from typing import List, Tuple
import tqdm
import datetime

import wandb
from pyserini.search import SimpleSearcher
from pyserini.search.lucene import LuceneSearcher

from retriever_utils import get_datasets, load_passages, validate, save_results
from options import print_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
#if logger.hasHandlers():
#    logger.handlers.clear()
#console = logging.StreamHandler()
#logger.addHandler(console)

RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.json"


class SparseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        index_name,
        use_rm3,
        num_threads,
        dedup=False
    ):  
        self.searcher = LuceneSearcher(index_name)
        # self.searcher = LuceneSearcher(f'{index_name}lucene9-index.cacm')
        logger.info(f"searcher.index_dir: {self.searcher.index_dir}") 
        self.use_rm3 = use_rm3
        ##### 启用rm3 QE方法
        if self.use_rm3:
            logger.info(f"Use rm3 QE.") 
            self.searcher.set_rm3()
        self.num_threads = num_threads
        self.dedup = dedup

    def get_top_docs(
        self, questions: List[str], top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
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
            logger.info(f"Deduplicate from {len(qids)} to {len(dedup_qids)}.")
            hits = self.searcher.batch_search(queries=dedup_questions, qids=dedup_qids, k=top_docs, threads=self.num_threads)
            for qid in dedup_map:
                hits[qid] = hits[dedup_map[qid]]
        else:
            hits = self.searcher.batch_search(queries=questions, qids=qids, k=top_docs, threads=self.num_threads)
        time1 = time.time()
        logger.info(f"Index search time: {time1 - time0} sec.")
        results = []
        for qid in qids:
            example_hits = hits[qid]
            example_top_docs = [hit.docid for hit in example_hits]
            example_scores = [hit.score for hit in example_hits]
            results.append((example_top_docs, example_scores))
        logger.info(f"Results conversion time: {time.time() - time1} sec.")
        return results


def main(args):
    config = vars(args)

    # get questions & answers
    qa_file_dict = get_datasets(args.qa_file)

    all_passages = load_passages(args.ctx_file)
    if len(all_passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )

    # Create or load retriever
    if args.pyserini_cache is not None:
        os.environ["PYSERINI_CACHE"] = args.pyserini_cache
    retriever = SparseRetriever(args.index_name, args.use_rm3, args.num_threads, args.dedup)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )

    # get top k results
    for dataset_name, (questions, question_answers) in tqdm.tqdm(qa_file_dict.items()):
        logger.info("*" * 40)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Current time {now_str}")
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        out_file = os.path.join(dataset_output_dir, RECALL_FILE_NAME)
        if os.path.exists(out_file):
            logger.info(f"Skipping dataset '{dataset_name}' as it already exists")
            continue
        os.makedirs(dataset_output_dir, exist_ok=True)

        top_ids_and_scores = retriever.get_top_docs(questions, args.n_docs)

        match_type = "regex" if "curated" in dataset_name else args.match
        # out_file = os.path.join(args.output_dir, RECALL_FILE_NAME)
        questions_doc_hits = validate(
            dataset_name,
            all_passages,
            question_answers,
            top_ids_and_scores,
            args.num_threads,
            match_type,
            out_file,
            use_wandb=use_wandb
        )

        out_file = os.path.join(dataset_output_dir, RESULTS_FILE_NAME)
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            out_file,
            output_no_text=args.output_no_text
        )

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qa_file",
        required=True,
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument(
        "--n-docs", type=int, default=100, help="Amount of top docs to return"
    )
    parser.add_argument("--output_no_text", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="wikipedia-dpr"
    )
    parser.add_argument(
        "--pyserini_cache",
        type=str,
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
    )
    # wandb params
    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_project",
        default="retrieval",
        type=str
    )
    parser.add_argument(
        "--wandb_name",
        default="spider-eval",
        type=str,
        help="Experiment name for W&B"
    )
    ##rm3 QE
    parser.add_argument(
        "--use_rm3",
        action="store_true",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert not os.path.exists(os.path.join(args.output_dir, RECALL_FILE_NAME))
    assert not os.path.exists(os.path.join(args.output_dir, RESULTS_FILE_NAME))
    print_args(args, args.output_dir)
    main(args)
    print('end!')

