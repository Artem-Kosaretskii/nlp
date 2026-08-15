import faiss
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_example(text, chunk_size, chunk_overlap, separators):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        print(f"chunk {i + 1}: {chunk}\n")
    return chunks


def vector_store_conf(chunks, metadata):

    docs = [
        Document(
            page_content=c,
            metadata={"source": m},
        ) for c, m in zip(chunks, metadata)
    ]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(docs)
    return vector_store


def main():
    load_dotenv()
    text = """Setting up a PostgreSQL server involves several key steps:
        1. Initializing the database cluster with the 'initdb' command.
        2. Configuring parameters in postgresql.conf (shared_buffers, max_connections).
        3. Configuring authentication in pg_hba.conf.
        4. Starting the server via systemd or pg_ctl.
        For optimal performance, it is recommended to allocate 25% of RAM to shared_buffers.
        If connection errors occur, check the logs in /var/log/postgresql/.
        Important: Create backups before changing any configuration files."""
    chunks1 = split_example(text, chunk_size=100, chunk_overlap=20, separators=["\n"])

    text = """Deep learning in medical diagnostics.
        Introduction. Modern convolutional neural networks achieve high accuracy in analyzing X-ray images.
        Methods. We compared ResNet-50 and Vision Transformer on the CheXpert dataset.
        Results. ViT showed superiority in detecting pneumonia.
        Discussion. Despite progress, challenges remain:
        1) Lack of labeled data.
        2) "Black-box" decision making.
        Conclusion. A promising direction is..."""
    chunks2 = split_example(text, chunk_size=150, chunk_overlap=30, separators=["\n\n", "\n", ". ", " "])

    vector_store = vector_store_conf(
        chunks=["Setting up a PostgreSQL server involves several key steps:",
                ": Important: Make backup copies before changing configs.",
                "To configure postgres, you need to..."],
        metadata=["a", "a", "b"]
    )
    result = vector_store.similarity_search("how to configure postgres", k=1, filter={"source": {"$eq": "a"}})

    vector_store = vector_store_conf(
        chunks=["RAG allows you to expand your LLM knowledge",
                "The cross-encoder allows ranking of the selected results",
                "RAG = robust attack on giants"],
        metadata=["RAG", "ML", "fantasy"]
    )
    result = vector_store.similarity_search("Why RAG is needed?", k=2, filter={"source": {"$neq": "fantasy"}})
    print(result)


if __name__ == "__main__":
    main()
