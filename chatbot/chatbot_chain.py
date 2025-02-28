import os
import time

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from vector_database import create_vector_database

load_dotenv()


def create_chatbot_chain():
    pipe = pipeline(
        "text-generation",
        model="kalisai/Nusantara-0.8b-Indo-Chat",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.3,
        repetition_penalty=1.1,
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=hf).bind(skip_prompt=True)

    retriever = create_vector_database()

    qa_system_prompt = (
        "Anda adalah asisten chatbot WhatsApp personal. "
        "Anda harus menjawab pertanyaan dengan akurat dan relevan berdasarkan konteks. "
        "Jika dalam konteks tidak ada informasi yang diperlukan, "
        "nyatakan dengan jelas bahwa Anda tidak ada informasi tersebut. "
        "Konteks: \n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

    # TODO: handle chat history
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


if __name__ == "__main__":
    chain = create_chatbot_chain()
    while True:
        message = input("Enter a message (or 'exit' to stop program): ")
        if message == "exit":
            break
        starttime = time.perf_counter_ns()
        res = chain.invoke({"input": message})
        endtime = time.perf_counter_ns()
        totaltime = (endtime - starttime) / 1_000_000_000
        # print(res)
        print(res["answer"])
        print(f"Total time: {totaltime:.3f}s")
