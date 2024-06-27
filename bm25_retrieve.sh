export JAVA_HOME=/path-to-java/
export JVM_PATH=/path-to-java/libjvm.so

python sparse_retriever.py \
--index_name "index_name" \
--qa_file "$qa_file" \
--ctx_file /path-to-Wikipedia_split/psgs_w100.tsv \
--output_dir $output_dir \
--n-docs 100 \
--num_threads 32 \
--no_wandb \
--dedup \
--pyserini_cache ./bm25_cache
