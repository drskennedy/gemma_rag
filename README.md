# Gemma 2B in a RAG Setup with EnsembleRetriever

**Step-by-step guide on Medium**: [Evaluating Gemma 2B in a RAG Setup with Basic and Advanced Retrievers](https://medium.com/@heelara/evaluating-gemma-2b-in-a-rag-setup-with-basic-and-advanced-retrievers-156ad26d56af)
___
## Context
Retrieval-Augmented Generation (RAG) is a popular technique used to improve the text generation capability of an LLM by keeping it fact driven and reduce its hallucinations. RAG performance is directly influenced by the embeddings formed from the chosen documents.
In this project, we will utilize Gemma 2B, a model introduced recently by Google, especially for consumer grade machines. We will test this LLM in a RAG setup for question-answering against a document that the model has not seen during its training. Its performance will be measured in terms of response accuracy, response time, memory and CPU utilization. The model was testing against 10 questions in file sh_qa_list.txt. Gemma's performance was compared against TinyLlama 1.1B, which is another LLM aimed at resource-constrained systems.
<br><br>
![System Design](/assets/gemma_rag_architecture.png)
___
## How to Install
- Create and activate the environment:
```
$ python3.10 -m venv mychat
$ source mychat/bin/activate
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download gemma-2b-it-q4_k_m.gguf from [lmstudio-ai HF repo](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/tree/main) to directory `./models/gemma_2b/`.
- Run script `main.py` to start the testing:
```
$ python main.py
```
- Optional: To launch alternate app of TinyLlama, run `main_tinyllama.py` to start its testing:
```
$ python main_tinyllama.py
```
___
## Quickstart
- To start the app using Gemma 2B, launch terminal from the project directory and run the following command:
```
$ source mychat/bin/activate
$ python main.py
```
- Here is a sample run:
```
$ python main.py
model;question;cosine;resp_time;memory_util;cpu_util
gemma_2b_en;Q1;0.47768;26.90;1.33;223.12
gemma_2b_en;Q2;0.70967;38.52;2.00;355.59
gemma_2b_en;Q3;0.86558;34.16;2.04;404.43
gemma_2b_en;Q4;0.86982;35.71;2.10;377.36
gemma_2b_en;Q5;0.71949;43.52;2.16;387.16
gemma_2b_en;Q6;0.77121;37.03;2.08;390.14
gemma_2b_en;Q7;0.37128;33.68;2.03;382.55
gemma_2b_en;Q8;0.69925;37.00;2.07;383.21
gemma_2b_en;Q9;0.52909;37.12;2.11;387.30
gemma_2b_en;Q10;0.68425;37.40;2.14;390.43
```
The results of the model evaluation as compared to TinyLlama in terms of cosine similarity of the responses captured in the following Treemap chart:
<br><br>
![Cosine Similarity Treemap](/assets/gemma_cosine.png)
___
## Key Libraries
- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Open-source library for efficient similarity search and clustering of dense vectors.
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Open-source pre-trained transformer model for embedding text to a dense vector space for tasks like cosine similarity calculation.

___
## Files and Content
- `models`: Directory hosting the downloaded LLM in GGUF format
- `opdf_index`: Directory for FAISS index and vectorstore
- `main.py`: Main Python module to launch the application using Gemma
- `main_tinyllama.py`: Alternate main Python module to launch the application using TinyLlama
- `LoadFVectorize.py`: Python module to load a pdf document, split and vectorize
- `LLMPerfMonitor.py`: Python module to return list of questions/answers, compute cosine similarity and compute memory/CPU stats
- `sh_qa_list.txt`: list of sample questions and answers
- `requirements.txt`: List of Python dependencies (and version)
___

## References
- [Gemma: Open Models Based on Gemini Research and Technology](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf), Google DeepMind.
- https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF
