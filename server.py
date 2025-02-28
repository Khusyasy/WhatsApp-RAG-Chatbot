# based on sample app https://developers.facebook.com/docs/whatsapp/sample-app-endpoints
import json
import logging
import os

import ngrok
import requests
from dotenv import load_dotenv
from flask import Flask, abort, request

load_dotenv()
PORT = int(os.environ["PORT"])
GRAPH_API_TOKEN = os.environ["GRAPH_API_TOKEN"]
WEBHOOK_VERIFY_TOKEN = os.environ["WEBHOOK_VERIFY_TOKEN"]

# ngrok for dev, easier to use
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", None)
logging.basicConfig(level=logging.DEBUG)
listener = ngrok.forward(PORT, domain=NGROK_DOMAIN, authtoken_from_env=True)

app = Flask(__name__)


@app.post("/webhook")
def handle_message():
    # https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#text-messages
    data = request.get_json()
    # print(json.dumps(data, indent=2))

    message = None
    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
    except KeyError as e:
        print("Error: ", e)
        return "OK"
    # print(json.dumps(message, indent=2))

    if message.get("type") == "text":
        phone_number_id = None
        try:
            phone_number_id = data["entry"][0]["changes"][0]["value"]["metadata"][
                "phone_number_id"
            ]
        except KeyError as e:
            print("Error: ", e)
            return "OK"
        print(f"phone_number_id: {phone_number_id}")

        # https://developers.facebook.com/docs/whatsapp/cloud-api/reference/messages
        reply_data = {
            "messaging_product": "whatsapp",
            "to": message["from"],
            "text": {"body": "Echo: " + message["text"]["body"]},
            "context": {
                "message_id": message["id"],
            },
        }
        requests.post(
            f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {GRAPH_API_TOKEN}"},
            json=reply_data,
        )

        # mark incoming message as read
        requests.post(
            f"https://graph.facebook.com/v18.0/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {GRAPH_API_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message["id"],
            },
        )

    return "OK"


@app.get("/webhook")
def handle_webhook():
    # https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests
    mode = request.args.get("hub.mode", "")
    token = request.args.get("hub.verify_token", "")
    challenge = request.args.get("hub.challenge", "")

    if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
        print("Webhook verified successfully!")
        return challenge

    print("Webhook verification failed!")
    abort(403)


@app.route("/")
def hello_world():
    return "<p>Nothing to see here.</p>"


if __name__ == "__main__":
    app.run(port=PORT)
