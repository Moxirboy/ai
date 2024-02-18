from flask import Flask, request, jsonify
from openai import OpenAI
import shelve
import os


OPEN_AI_API_KEY = "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"
client = OpenAI(api_key=OPEN_AI_API_KEY)
app = Flask(__name__)
##  threads here
def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)

def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id



def generate_response(message_body, wa_id, name):
    ## wa_id - id for thread
    thread_id = check_if_thread_exists(wa_id)

    # create thread
    if thread_id is None:
        print(f"Creating new thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id

    # existing thred
    else:
        print(f"Retrieving existing thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",   # always user, d0n't change, postgre dan usernamela tiqiladi shu yerga
        content=message_body
    )

    ## call for run assistant
    new_message = run_assistant(thread)
    print(f"To {name}:", new_message)
    return new_message


def run_assistant(thread):
    # call for assistant 
    assistant = client.beta.assistants.retrieve("asst_3S1rlXEXttRHfMAkqIfewd2M")

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    ### logs
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # message manipulation
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    print(f"Generated message: {new_message}")
    return new_message