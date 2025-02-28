# WhatsApp RAG Chatbot

- Setup [WhatsApp Api](https://developers.facebook.com/docs/whatsapp/cloud-api/get-started)

## Chatbot

Scrape data from Wikipedia, update the `FIRST_URL`, `KEYWORD`, etc. in the file first.
```bash
python chatbot/scraping_wikipedia.py
```

Create vector database.
```bash
python chatbot/vector_database.py
```

Run the chatbot locally.
```bash
python chatbot/chatbot_chain.py
```


## Server

```bash
python server.py
```
