import os
import json
import uuid

BASE_PATH = "data/conversations"


# -----------------------------------
# ensure folder exists
# -----------------------------------

os.makedirs(BASE_PATH, exist_ok=True)


# -----------------------------------
# create new conversation
# -----------------------------------

def create_conversation():

    conv_id = str(uuid.uuid4())

    data = {
        "id": conv_id,
        "title": "New Conversation",
        "messages": []
    }

    save_conversation(conv_id, data)

    return conv_id


# -----------------------------------
# save conversation
# -----------------------------------

def save_conversation(conv_id, data):

    path = os.path.join(BASE_PATH, f"{conv_id}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------------
# load conversation
# -----------------------------------

def load_conversation(conv_id):

    path = os.path.join(BASE_PATH, f"{conv_id}.json")

    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------
# list conversations
# -----------------------------------

def list_conversations():

    conversations = []

    for file in os.listdir(BASE_PATH):

        if file.endswith(".json"):

            path = os.path.join(BASE_PATH, file)

            with open(path, "r") as f:

                data = json.load(f)

                conversations.append({
                    "id": data["id"],
                    "title": data["title"]
                })

    return conversations