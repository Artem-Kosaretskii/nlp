import os
import torch
from torch import Tensor
import torch.nn.functional as F
import faiss
import numpy as np
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List, Optional
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


class RAG:

    def __init__(self, embedder_name: str, reranker_name: str, chunk_size: int = 500, chunk_overlap: int = 125, device: Optional[str] = 'cuda'):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        self.embedder = AutoModel.from_pretrained(embedder_name).to(self.device)
        self.embedder.eval()
        self.rr_tokenizer = AutoTokenizer.from_pretrained(reranker_name, padding_side='left')
        self.reranker = AutoModelForCausalLM.from_pretrained(reranker_name).to(self.device)
        self.reranker.eval()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index = None
        self.doc_store = []
        self.max_length = 8192
        self.token_false_id = self.rr_tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.rr_tokenizer.convert_tokens_to_ids("yes")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.rr_tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.rr_tokenizer.encode(suffix, add_special_tokens=False)

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        inputs = self.emb_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.embedder(**inputs)
        inputs.to("cpu")
        embeddings = self.last_token_pool(outputs.last_hidden_state, inputs.attention_mask).cpu()
        return F.normalize(embeddings, p=2, dim=1).float().numpy()

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device),sequence_lengths]

    def load_and_process_file(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1]
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

    def build_index(self, file_paths: List[str], batch_size: int = 32) -> None:
        all_docs = []
        for path in file_paths:
            all_docs.extend(self.load_and_process_file(path))
        self.doc_store = all_docs
        embeddings = []
        for i in range(0, len(all_docs), batch_size):
            batch = [doc.page_content for doc in all_docs[i:i + batch_size]]
            embeddings.append(self._generate_embeddings(batch))
        embeddings = np.concatenate(embeddings)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str):
        return f'Instruct: {task_description}\nQuery:{query}'

    @staticmethod
    def format_reranker_instruction(query, doc, instruction=None):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def process_inputs(self, pairs):
        inputs = self.rr_tokenizer(pairs,
                                   padding=False,
                                   truncation='longest_first',
                                   return_attention_mask=False,
                                   max_length=self.max_length -
                                   len(self.prefix_tokens) -
                                   len(self.suffix_tokens))
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.rr_tokenizer.pad(inputs,
                                       padding=True,
                                       return_tensors="pt",
                                       max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def search(self, query: str,  k: int = 5, task: str = None):
        if self.index is None:
            raise ValueError("Index not initialized")
        if task is None:
            task = 'Given a web search query, retrieve relevant passages that answer the query'
        query_embedding = self._generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    @torch.no_grad()
    def compute_logits(self, inputs):
        batch_scores = self.reranker(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, query: str, documents: List[str], batch_size=1):
        pairs = []
        for d in documents:
            pairs.append(self.format_reranker_instruction(query, d))

        scores = []
        for i in range(0, len(pairs), batch_size):
            inputs = self.process_inputs(pairs[i:i + batch_size])
            sc = self.compute_logits(inputs)
            scores.extend(sc)
        return scores


def main():


    load_dotenv()
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    rag = RAG(model_name, model_name)
    q = "what is the attention mechanism used in deepseek-v3"
    rag.build_index([
        "data/deepseek_r1.pdf",
        "data/deepseek_v3_tech_report.pdf",
        "data/gemini_1.5.txt"
    ])
    k = 100
    D, I = rag.search(q, k=k)
    candidates = [rag.doc_store[i].page_content for i in I[0]]

    k = 5
    scores = rag.rerank(q, candidates)
    scores_array = np.array(scores)
    indices = np.argsort(scores_array)[::-1][:k]

    print('Top candidates:')
    for i in indices:
        print(candidates[i])
        print("-#" * 20)
        print()

    torch.cuda.empty_cache()
    rag.embedder.to("cpu")
    rag.reranker.to("cpu")
    prompt = f"Given texts info below give me a very short answer to a question: {q}\n\n"

    for i, v in enumerate(indices):
        prompt += f"chunk {i}: {candidates[v]}\n\n"

    torch.manual_seed(42)
    llm_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto").to("cuda")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("Answer:", content.split('</think>')[1].strip('\n'))


if __name__ == '__main__':
    main()
