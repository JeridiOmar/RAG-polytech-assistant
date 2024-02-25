from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, ServiceContext, load_index_from_storage, StorageContext, Prompt
import os.path

# query_wrapper_prompt = PromptTemplate(
#     "voici des informations qui vont t'aider à repondre avec ce contexte. \n "
#     "{context_str} \n"
#     "avec ces informations reponds à cette question. Question :{query_str}\n "
# )
# prompt(
#     "We have provided context information below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Do not give me an answer if it is not mentioned in the context as a fact. \n"
#     "Given this information, please provide me with an answer to the following:\n{query_str}\n"
# )
storage_path = "/home/pred_index_23/llama-index-storage"

docs_path = "/home/pred_index_23/llama-index-data"
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    # query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)
# Settings.chunk_size = 512
Settings.llm = llm
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local")

# if not os.path.exists(storage_path):
if True:
    print("starting reading data")
    documents = SimpleDirectoryReader(docs_path).load_data()
    print("starting indexing data")
    print(documents[0].get_content())

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=storage_path)

    query_engine = index.as_query_engine(service_context=service_context)
else:
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_path),
                                    service_context=service_context)
    query_engine = index.as_query_engine(service_context=service_context)

while 1:

    documents[0].get_content()

    req = input("c'est quoi ton question: ")
    response = query_engine.query(req)
    print(response)
