import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import gradio as gr

DOCS_PATH = './docs'
INDEX_PATH = './faiss_index'
EMBEDDING_MODEL = 'jhgan/ko-sbert-nli'
LLM_MODEL_PATH = os.getenv('LLM_MODEL_PATH', './models/EEVE-Korean-10.8B.gguf')


def load_or_create_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = DirectoryLoader(DOCS_PATH, recursive=True, show_progress=True)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_PATH)
    return vectorstore


def create_qa_chain():
    vectorstore = load_or_create_vectorstore()
    llm = LlamaCpp(model_path=LLM_MODEL_PATH, n_gpu_layers=0, verbose=False)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
    return qa_chain


def answer_question(question, chain=None):
    if chain is None:
        chain = create_qa_chain()
    result = chain.run(question)
    return result


if __name__ == '__main__':
    qa_chain = create_qa_chain()
    iface = gr.Interface(
        fn=lambda q: answer_question(q, qa_chain),
        inputs=gr.Textbox(label='질문을 입력하세요'),
        outputs=gr.Textbox(label='답변'),
        title='로컬 문서 기반 RAG Q&A',
        description='docs 폴더의 문서들을 기반으로 질문에 답변합니다.'
    )
    iface.launch()
