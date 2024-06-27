
import json
import sys
import tqdm
import glob
import random

def process_worker(results_dir):
    with open(results_dir, 'r') as f:
        data = json.load(f)
        key = results_dir.split('/')[-2]
        results = []
        ctxs = []
        random.shuffle(data)
        for d in data[0:30]:
            results.append([d['question'], d['hit_min_rank'], " ".join([d["ctxs"][0]["title"],d["ctxs"][1]["title"],d["ctxs"][2]["title"]]), " ".join([d["ctxs"][0]["text"],d["ctxs"][1]["text"],d["ctxs"][2]["text"]])])
    return key, results


def main():
    if len(sys.argv) < 5:
        print('Usage: python retrieval_result_synthesize.py <output_file> <results_dir_glob pattern> <n_workers> <n_examples>')
        exit(0)

    output_file = sys.argv[1]
    n_examples = int(sys.argv[4])
    results_dirs = [sys.argv[2]%(i) for i in range(n_examples)] 
    n_workers = int(sys.argv[3])
    all_data = {}
    if n_workers == 1:
        for results_dir in tqdm.tqdm(results_dirs):
            with open(results_dir, 'r') as f:
                data = json.load(f)
                key = results_dir.split('/')[-2]
                results = []
                ctxs = []
                random.shuffle(data)
                for d in data[0:30]:
                    results.append([d['question'], d['hit_min_rank'], " ".join([d["ctxs"][0]["title"],d["ctxs"][1]["title"],d["ctxs"][2]["title"]]), " ".join([d["ctxs"][0]["text"],d["ctxs"][1]["text"],d["ctxs"][2]["text"]])])
                all_data[key] = results
    else:
        import multiprocessing as mp
        pool = mp.Pool(n_workers)
        all_data_list = pool.map(process_worker, results_dirs)

        pool.close()
        pool.join()
        all_data = dict(all_data_list)
    
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

if __name__ == '__main__':
    main()


