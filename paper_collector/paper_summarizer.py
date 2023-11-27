import time 
import os
import gc
import re
import base64
import textwrap
import torch
import evaluate
import transformers

import numpy as np
import pandas as pd
import streamlit as st
import arxiv as arx

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from utils.custom_utils import save_var, load_var

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft import LoraModel, PeftModel

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from langchain.chains import ConversationChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


class Summarizer():
    
    def __init__(self):
        self.summarization_results = []
        self.load_llm()

    def empty_mem(self):
        torch.cuda.empty_cache()
        return gc.collect()

    def load_llm(self):
        
        self.empty_mem()

        # temporary model for dev.
        self.model_id = "farleyknight-org-username/arxiv-summarization-t5-small"

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, 
            # quantization_config=bnb_config, 
            device_map="auto",
        )

        # adapter_name = './adapters/'

        # self.lora_config = LoraConfig.from_pretrained(adapter_name)
        # self.model = get_peft_model(self.model, self.lora_config)

        self.pipeline_base = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task='summarization',
            repetition_penalty=1.2,
            temperature=0.1,
            do_sample=True,
            min_length=100,
            max_length=500,
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipeline_base)
        
    def prepare_documents(self, docs):
        docs = [Document(page_content=t) for t in docs]
        return docs
    
    def set_summary(self, summary):
        self.summary = summary
        
    def get_summary(self):
        return self.summary_to_text(self.summary)
        
    def summary_to_text(self, summary):
        return '\n\n'.join(summary)

    def summarize(self, docs, verbose=False):

        # Define prompt
        self.prompt_template = """Write concepts of the following:
"{text}"
CONCEPTS:"""

        prompt = PromptTemplate.from_template(self.prompt_template)

        # Define LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, 
            document_variable_name="text",
            verbose=verbose,
        )
        
        prepared_docs = self.prepare_documents(docs)

        for doc in tqdm(prepared_docs):
            self.summarization_results.append(stuff_chain.run([doc]))
            
        self.set_summary(self.summarization_results)
        self.empty_mem()