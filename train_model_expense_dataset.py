# pip install openai langchain 

import json 
import time 
# from langchain.schema import AIMessage
# from langchain.adapters.openai import convert_message_to_dict
import openai
from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

# os.environ["OPENAI_API_KEY"] = "="

if __name__=="__main__":

    my_file = "expense.json"

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
        suffix="expense"
    )

    ftj = client.fine_tuning.jobs.retrieve(job.id)