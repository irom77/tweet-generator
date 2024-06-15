# pip install openai langchain streamlit streamlit_feedback

import json 
import time 
from langchain.schema import AIMessage
from langchain.adapters.openai import convert_message_to_dict
import openai
from openai import OpenAI
import os 
from dotenv import dotenv_values

os.environ["OPENAI_API_KEY"] = ""

if __name__=="__main__":

    # prepare the data into fine-tune format

    with open('dataset_elon_tweets.json') as f:
        data = json.load(f)


    tweets = [ d["full_text"] for d in data if "t.co" not in d["full_text"]]
    # print(tweets)

    messages = [AIMessage(content=t) for t in tweets]
    # print(messages)

    system_message = {"role": "system", "content": "write a tweet"}
    data = [[system_message, convert_message_to_dict(m)] for m in messages]

    # print(data)

    # prepare a fine-tuning file

    my_file = "dataset_elon_tweets_training.json"

    with open(my_file, 'w', encoding='utf-8') as file:
        for m in data:
            file.write(json.dumps({"messages": m}) + "\n")

    # upload the fine-tune file

    client = OpenAI()

    training_file = client.files.create(
        file=open(my_file, "rb"),
        purpose="fine-tune"
    )

    # create a fine-tune model job

    job = client.fine_tuning.jobs.create(
        training_file = training_file.id,
        model = "gpt-3.5-turbo",
        suffix="elon-twitter"
    )

    ftj = client.fine_tuning.jobs.retrieve(job.id)

    






