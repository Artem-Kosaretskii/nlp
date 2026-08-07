import numpy as np
import torch
from dotenv import load_dotenv
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class RAG:

    def __init__(self, llm_name: str, embedder_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        self.embedder_model = AutoModel.from_pretrained(embedder_name).to(self.device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_name).to(self.device)
        self.knowledge_base = {"texts": [], "embeddings": np.array([])}

    def get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.embedder_tokenizer(text,
                                         return_tensors="pt",
                                         padding="max_length",
                                         truncation=True,
                                         max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.embedder_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def add_to_knowledge_base(self, text: str | List[str]) -> None:

        embedding = self.get_embedding(text).cpu().numpy()

        if isinstance(text, str):
            self.knowledge_base["texts"].append(text)
        else:
            self.knowledge_base["texts"].extend(text)

        if self.knowledge_base["embeddings"].shape[0] == 0:
            self.knowledge_base["embeddings"] = embedding
        else:
            self.knowledge_base["embeddings"] = np.vstack([self.knowledge_base["embeddings"], embedding], dim=0)

    def find_closest(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:

        if len(self.knowledge_base["texts"]) == 0:
            return []
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.knowledge_base["embeddings"])[0]
        top_indices = similarities.argsort().argsort()[::-1][:min(top_k, len(similarities))]
        results = []
        for idx in top_indices:
            results.append({"text": self.knowledge_base["texts"][idx], "score": similarities[idx]})
        return results

    def _create_prompt(self, question: str, context_texts: List[str]) -> str:
        context = "\n\n".join([f"Context {i + 1}: {text}" for i, text in enumerate(context_texts)])
        prompt = f"""Using the contexts below, answer the question as briefly as possible. If the contexts don't contain the information you need, say so.
        {context}
        Question: {question}"""
        return prompt

    def ask_question(self, question: str, top_k: int = 3) -> str:
        question_embedding = self.get_embedding(question)
        closest = self.find_closest(question_embedding.cpu().numpy(), top_k=top_k)
        if len(closest) > 0:
            context_texts = [item["text"] for item in closest]
        else:
            context_texts = ["Nothing relevant has been found"]

        prompt = self._create_prompt(question, context_texts)
        messages = [{
            "role":
                "system",
            "content":
                "You are a virtual assistant. Your job is to be a helpful conversational assistant."
        }, {
            "role": "user",
            "content": prompt
        }]

        text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.llm_model.generate(**model_inputs,
                                                max_new_tokens=1024,
                                                do_sample=True,
                                                temperature=0.1)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def main():
    load_dotenv()
    llm_id = 'Qwen/Qwen2.5-0.5B-Instruct'  # "t-tech/T-lite-it-1.0"
    embedder_id = 'cointegrated/rubert-tiny2'
    rag = RAG(llm_name=llm_id, embedder_name=embedder_id)
    db = ["The password to my computer is X0Ja_asd", "RisingTide is a new group, consisting of former sailors"]
    rag.add_to_knowledge_base(db)
    answer = rag.ask_question("I have forgotten a password to my computer")
    print(answer)

if __name__ == '__main__':
    main()
