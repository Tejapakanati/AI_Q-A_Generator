import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

from src.prompt import prompt_template, refine_template


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from your .env file")


def file_processing(file_path):
    # Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine PDF page content
    question_gen = ""

    for page in data:
        question_gen += page.page_content + "\n"

    # Split text for question generation
    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunk_ques_data = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [
        Document(page_content=text)
        for text in chunk_ques_data
    ]

    # Split documents for answer generation
    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=500,
        chunk_overlap=50,
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    # Process the PDF
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # LLM for question generation
    llm_ques_gen_pipeline = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
    )

    # Question generation prompts
    prompt_questions = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
    )

    refine_prompt_questions = PromptTemplate(
        template=refine_template,
        input_variables=["existing_answer", "text"],
    )

    # Generate questions
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=prompt_questions,
        refine_prompt=refine_prompt_questions,
    )

    ques = ques_gen_chain.run(document_ques_gen)

    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        document_answer_gen,
        embeddings,
    )

    # LLM for answering questions
    llm_answer_gen = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
    )

    # Filter generated questions
    ques_list = ques.split("\n")

    filtered_quest_list = [
        q.strip()
        for q in ques_list
        if q.strip().endswith(("?", "."))
    ]

    # Create retrieval-based answer chain
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    return answer_generation_chain, filtered_quest_list

