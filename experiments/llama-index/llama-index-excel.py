from pathlib import Path
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, ServiceContext, load_index_from_storage, StorageContext, Prompt

storage_path = "/home/pred_index_23/llama-index-storage/excel"
sheet_path = "/home/pred_index_23/llama-index-data/exportMaquette_MAT_2023_valid_en.xls"

PandasExcelReader = download_loader("PandasExcelReader")
loader = PandasExcelReader()
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
documents = loader.load_data(file=Path(sheet_path), sheet_index=None)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)
while 1:
    req = input("c'est quoi ta question: ")
    response = query_engine.query(req)
    print(response)
