from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredMarkdownLoader

path = "data"
mode = "pdf"
chunk_size=800


if mode == "pdf":
    glob = "*.pdf"
    loader = DirectoryLoader(path=path, glob=glob, loader_cls=PyPDFLoader, 
                             use_multithreading=True)
else:
    glob = "*.md"
    loader = DirectoryLoader(path=path, glob=glob, use_multithreading=True)
    

documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss")
