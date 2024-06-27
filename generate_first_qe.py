import tqdm
import os
import re
import random
import json
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


model_path = '/path-to-model/Mistral-7B-Instruct-v0.2'

llm_base = LLM(model_path, tensor_parallel_size=2)

question_file = 'path-to-input/nq-test.csv'
output_file = 'path-to-output/first-qe.txt'
print(f"Start generate: {question_file}\n")
with open(output_file, 'a') as fw:
    data = open(question_file, 'r').readlines()
    for i in tqdm.trange(0,len(data)):
        q_base = data[i].split('\t')[0]
        print(q_base)
        instruction_kp_analyze = """[INST]Please note that this is a brand new conversation start. When responding to the following questions, disregard all previous interactions and context.
Question: {query}

When analyzing a phrase, first consider if the phrase could be a proper noun, such as the title of a song, movie, book, or other work. If it is a common phrase or doesn't immediately appear to be a title, then proceed to analyze its grammatical structure as a standard phrase. However, if there is a possibility that the phrase is a title, treat it as a proper noun and analyze it in that context.

Do not attempt to explain or answer the question, just provide the key phrases.

Expected Output:
"Key Phrases Output": key phrases in "{query}"

Output:[/INST]""".format(query=q_base.strip())
        sampling_params = SamplingParams(temperature=0.2, max_tokens=150, repetition_penalty=1.1)
        output = llm_base.generate(instruction_kp_analyze,sampling_params)
        answer_kp_analysis = output[0].outputs[0].text
        print(answer_kp_analysis)
        answer_kp_analysis = answer_kp_analysis.replace('\n',' ').strip()

        instruction_analyze = """[INST]Question: {query}
Key Phrases in query:{answer_kp_analysis}

Analyze the question carefully. Determine what type of information is being asked for. Consider the most direct way to find this information. If the question is about identifying something or someone, focus on the specific details provided. Avoid assumptions or interpretations beyond what is explicitly asked. Provide a clear and concise answer based on the analysis.

Do not attempt to explain or answer the question, just provide the Question Analysis.

Expected Output:
"Question Analysis": Question Analysis based on Key Phrases

Output:[/INST]""".format(query=q_base.strip(),answer_kp_analysis=answer_kp_analysis)
        sampling_params = SamplingParams(temperature=0.2, max_tokens=150, repetition_penalty=1.1)
        output = llm_base.generate(instruction_analyze,sampling_params)
        answer_analysis = output[0].outputs[0].text
        print(answer_analysis)
        answer_analysis = answer_analysis.replace('\n',' ').strip()
        
        instruction_extend = """[INST]Question: {query}
Question analysis: {answer_analysis}

Based on the analysis and your available knowledge, create a possibly correct and concise answer that directly answers the question "{query}".

Expected Output:
"Answer": answer with a detailed context
Output:
"Answer":[/INST]""".format(query=q_base.strip(), answer_analysis=answer_analysis)
        sampling_params = SamplingParams(temperature=0.8, max_tokens=100, repetition_penalty=1.1,n=30)
        output = llm_base.generate(instruction_extend,sampling_params)
        result_extend = []
        result_extend += [output[0].outputs[i].text.replace('\n',' ').strip().strip('Answer: ') for i in range(10)]
        random.shuffle(result_extend)
        print(result_extend)
        
        for it in result_extend:
            item = [str(i), it.replace('\n',' ').strip()]
            fw.write('\t'.join(item) + '\n')
            fw.flush()


