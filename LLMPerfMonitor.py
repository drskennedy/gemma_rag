# LLMPermMonitor.py

from sentence_transformers import SentenceTransformer, util
import os, psutil

import nltk
from nltk.translate import meteor
from nltk import word_tokenize
#nltk.download('punkt')
#nltk.download('wordnet')

def get_questions_answers() -> list[str]:
    # returns a list of questions interleaved with answers
    with open("sh_qa_list.txt") as qfile:
        lines = [line.rstrip()[3:] for line in qfile]
    return lines

def calc_similarity(sent1,sent2) -> float:
    # creates embeddings, computes cosine similarity and returns the value
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #Compute embedding for both strings
    embedding_1= model.encode(sent1, convert_to_tensor=True)
    embedding_2 = model.encode(sent2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2).item()

def get_mem_cpu_util(ppid) -> tuple[float,float]:
    # collects RSS in GB, total CPU util and returns them
    process = psutil.Process(ppid) # return parent process
    mem_usage = process.memory_info().rss / 1024 ** 3  # in GB
    cpu_usage = sum(psutil.cpu_percent(percpu=True))
    return mem_usage,cpu_usage
