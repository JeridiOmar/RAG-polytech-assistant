from pathlib import Path
import os
import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, ServiceContext, load_index_from_storage, StorageContext, Prompt, \
    VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getbuffer())
        temp_file.flush()

        # Process the PDF file
        with open(temp_file.name, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            num_pages = pdf_reader.numPages
            st.write(f"Number of pages in the PDF: {num_pages}")

        # Remove the temporary file
        os.unlink(temp_file.name)


def main(query):
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
    # loader = PdfReader('/home/pred_index_23/llama-index-data/Maquette_MAT_2023_en.pdf')
    # documents = loader.load_data(file=Path('/home/pred_index_23/llama-index-data/Maquette_MAT_2023_en.pdf'))
    loader = PyMuPDFReader()
    documents = loader.load_data('/home/pred_index_23/llama-index-data/Maquette_MAT_2023_en.pdf')

    QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question in bullet point format with a new line after each point and cross reference any data cited in the document.\n"
        "warn the user if any information seems off: {query_str}\n"
    )
    QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

    index = VectorStoreIndex(documents, service_context=service_context)
    query_engine = index.as_query_engine(service_context=service_context)

    # response = query_engine.query(query, text_qa_template=QA_PROMPT)
    response = query_engine.query(query)
    print(response)
    return response


user_input = input('Enter prompt:')
print(main(query=user_input))
