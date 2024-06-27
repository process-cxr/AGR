import tqdm
import os
import re
import random
import json
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

model_path = '/path-to-model/Mistral-7B-Instruct-v0.2'
llm = LLM(model_path, tensor_parallel_size=2)

candidate_docs_file = 'path-to-output/candidate_docs.json'
output_file = 'path-to-output/refine-qe.txt'
json_file = 'path-to-output/refine-qe.json'
record =[]
print(f"Start generate : {output_file}\n")
with open(output_file, 'a') as fw:
    data = json.load(open(candidate_docs_file, 'r'))
    for i in tqdm.trange(0,len(data)):
        result_extend = data[f'nq-{i}']
        q_base = result_extend[0][0].split('?')[0]
        candidate = [temp[0][len(q_base)+2:] for temp in result_extend]
        random.shuffle(result_extend)
        print(q_base)
        candidate_docs = "\n".join(
            [f"Reference context: {temp[3]}" for i, temp in enumerate(result_extend[0:10],1)]
        )
        print(candidate_docs)
        
        instruction_extend = """[INST]Question: {query}
Retrieval Context: {candidate_docs}

Based on the retrieval context and your available knowledge, create a possibly correct and concise answer that directly answers the question "{query}".

Expected Output:
"Answer": answer with a detailed context
Output:
"Answer":[/INST]""".format(query=q_base.strip(), candidate_docs=candidate_docs)
        
        sampling_params = SamplingParams(temperature=0.8, max_tokens=100, repetition_penalty=1.1,n=10)
        output = llm.generate(instruction_extend,sampling_params)
        result_extend = []
        result_extend += [output[0].outputs[i].text.replace('\n',' ').strip().strip('Answer: ') for i in range(10)]
        random.shuffle(result_extend)

        answer_qe_candidate = "\n".join(
            [f"Candidate Answer {str(i+1)}: {temp.strip()}" for i, temp in enumerate(result_extend)]
        )
        print(answer_qe_candidate)
                
        instruction_selectd = """[INST]Question: {query}
Candidate answer list:
{answer_list}

Based on the candidate answers, please evaluate the accuracy and reliability of each candidate answer. Identify any misinformation or incorrect facts in the answers. Please use all your available knowledge to verify the accuracy of these candidate answers. Then, generate a correct and concise response that best answer the question, refer to the information from the candidate answers that you have verified as accurate.

Expected Output:
"Best Answer": a concise answer for the question "{query}"
"Explanation": 
Output:
Best Answer: [/INST]""".format(query=q_base.strip(),answer_list=answer_qe_candidate)
        sampling_params = SamplingParams(temperature=0.2, max_tokens=300, repetition_penalty=1.1)
        output = llm.generate(instruction_selectd,sampling_params)
        answer_qe_refined = output[0].outputs[0].text.replace('\n',' ').strip().strip('Best Answer:')
        print(answer_qe_refined)
    
        match = re.search(r"(.*?)\s*Explanation:",answer_qe_refined,re.IGNORECASE)

        best_result = match.group(1) if match else None
        if best_result is not  None:
            print(f"Match: {best_result}\n")
            answer_qe_refined_detail = f"Match: {best_result}\n"
        else:
            random_integer = random.randint(0, 9)
            best_result = result_extend[random_integer]
            print(f"Not Match, sample answer: {best_result}\n")
            answer_qe_refined_detail = f"Not Match, sample answer: {best_result}\n"
            
        item = [str(i), best_result.replace('\n',' ').strip()]
        print('\t'.join(item) + '\n')
        fw.write('\t'.join(item) + '\n')
        fw.flush()
        temp_record = {
            'candidate_docs': candidate_docs,
            'answer_qe_candidate': answer_qe_candidate,
            'answer_qe_refined': answer_qe_refined_detail
        }
        record.append(temp_record)
json.dump(record, open(json_file, "w"), indent=2, ensure_ascii=False)
