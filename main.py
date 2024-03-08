# main_gemma.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
import LoadVectorize
import LLMPerfMonitor
from langchain.retrievers import EnsembleRetriever
import threading                                                                                                                          
import os
import time
import timeit

def monitor_thread(event, ppid, shared_list):
    while not event.is_set():
        mem,cpu = LLMPerfMonitor.get_mem_cpu_util(ppid)  # Run the async task in the thread's event loop
        shared_list += [mem,cpu]
        time.sleep(1)

def main():
    event = threading.Event()  # Create an event object
    shared_list = []  # Create a shared Queue object
    child = threading.Thread(target=monitor_thread, args=(event,os.getpid(),shared_list))
    child.start()

    db,bm25_r = LoadVectorize.load_db()
    faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}, max_tokens_limit=1000)
    r = 0.3
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_r,faiss_retriever],weights=[r,1-r])

    qa_list = LLMPerfMonitor.get_questions_answers()

    # Prompt template 
    qa_template = """
    <bos><start_of_turn>user
    {context}
    {question}<end_of_turn>
    <start_of_turn>model
    """

    # Create a prompt instance 
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    llm = LlamaCpp(
        model_path="./models/gemma_2b/gemma-2b-it-q4_k_m.gguf",
        temperature=0.01,
        max_tokens=2000,
        top_p=1,
        verbose=False,
        n_ctx=2048
    )
    # Custom QA Chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=ensemble_retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    print('model;question;cosine;resp_time;memory_util;cpu_util')
    for i,query in enumerate(qa_list[::2]):
        start = timeit.default_timer()
        result = qa_chain({"query": query})
        time = timeit.default_timer() - start # seconds
        avg_mem = sum(shared_list[::2])/len(shared_list[::2])
        avg_cpu = sum(shared_list[1::2])/len(shared_list[1::2])
        shared_list.clear()
        cos_sim = LLMPerfMonitor.calc_similarity(qa_list[i*2+1],result["result"])
        print(f'gemma_2b;Q{i+1};{cos_sim:.5};{time:.2f};{avg_mem:.2f};{avg_cpu:.2f}')
        #print(f'Q{i+1}: {query}\nA: {result["result"]}\nSME: {qa_list[i*2+1]}\n')

    event.set()  # Set the event to signal the child thread to terminate
    child.join()  # Wait for the child thread to finish

if __name__ == "__main__":
    main()
