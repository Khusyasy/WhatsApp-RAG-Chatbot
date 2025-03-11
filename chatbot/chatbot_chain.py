import time

from chat_history import get_session_history
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from vector_database import create_vector_database

load_dotenv()


def create_chatbot_chain():
    model_name = "kalisai/Nusantara-0.8b-Indo-Chat"
    # model_name = "kalisai/Nusantara-1.8b-Indo-Chat"
    pipe = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.3,
        repetition_penalty=1.1,
        device_map="auto",
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=hf).bind(skip_prompt=True)

    retriever = create_vector_database()

    # chain buat bikin ulang pertanyaan sesuai konteks
    contextualize_q_system_prompt = "Diberikan riwayat obrolan dan pertanyaan user terbaru yang mungkin merujuk pada konteks pada riwayat obrolan, rumuskan pertanyaan mandiri yang dapat dipahami tanpa riwayat obrolan. JANGAN menjawab pertanyaan tersebut, cukup rumuskan ulang jika diperlukan dan kembalikan apa adanya."
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history", optional=True, n_messages=6),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    # chain buat jawaban
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
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


if __name__ == "__main__":
    chain = create_chatbot_chain()
    while True:
        message = input("Enter a message (or 'exit' to stop program): ")
        if message == "exit":
            break
        starttime = time.perf_counter_ns()
        res = chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": "abc123"}},
        )
        endtime = time.perf_counter_ns()
        totaltime = (endtime - starttime) / 1_000_000_000
        # print(res)
        print(res["answer"])
        print(f"Total time: {totaltime:.3f}s")
