from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    history = SQLChatMessageHistory(
        session_id=session_id, connection="sqlite:///history.db"
    )
    return history
