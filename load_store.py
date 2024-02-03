from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

def load_llm():
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q2_K.bin', # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256, 
                            'temperature': 0, 
                            'context_length' : 2048})
    return llm

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    db = FAISS.load_local("faiss", embeddings)
    return db

def create_prompt_template():
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".
    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    return prompt

def create_qa_chain():
    """create the qa chain"""

    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store()
    prompt = create_prompt_template()


    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    
    return qa_chain

def generate_response(query, qa_chain):
    return qa_chain({'query': query})['result']

