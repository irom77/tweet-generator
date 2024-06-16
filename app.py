# pip install openai langchain streamlit streamlit_feedback langsmith langchain-openai
# make sure to create an api in langchain smith console

import os, sys
import langchain
import streamlit as st
from langsmith import Client
from streamlit_feedback import streamlit_feedback

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

load_dotenv()

client = Client(api_key=os.environ["api_key"])

# os.environ["OPENAI_API_KEY"] = ""
sys.exit

if "run_id" not in st.session_state:
    st.session_state.run_id = None

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

normal_chain = (
    ChatPromptTemplate.from_messages([("system", "write a tweet about {topic} in the style of Elon Musk") ])
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    ChatPromptTemplate.from_messages([("system", "write a tweet about {topic}") ])
    | ChatOpenAI(model="ft:gpt-3.5-turbo-0125:personal:elon-twitter:9aW32rvB")
    | StrOutputParser()
)

def generate_tweet_normal(topic):
    result = normal_chain.invoke({"topic": topic})
    wait_for_all_tracers()
    return result

def generate_tweet(topic):
    result = chain.invoke({"topic": topic}, config=runnable_config)
    run = run_collector.traced_runs[0]
    run_collector.traced_runs = []
    st.session_state.run_id = run.id
    wait_for_all_tracers()
    return result


col1, col2 = st.columns([1, 6])  # Adjust the ratio for desired layout

col2.title("Elon Musk Tweet Generator")

st.info("\n\nTwo tweets will be generated: one using a finetuned model, one using a prompted model. Afterwards, you can provide feedback about whether the finetuned model performed better!")

topic = st.text_input("Enter a topic:")
if 'show_tweets' not in st.session_state:
    st.session_state.show_tweets = None

if st.button("Generate Tweets"):
    if topic:
        col3, col4 = st.columns([6, 6])
        tweet = generate_tweet(topic)
        col3.markdown("### Finetuned Tweet:")
        col3.write(f"🐦: {tweet}")
        col3.markdown("---")  # Add a horizontal line for separation
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            key=f"feedback_{st.session_state.run_id}",
            align="flex-start"
        )
        scores = {"👍": 1, "👎": 0}
        if feedback:
            score = scores[feedback["score"]]
            feedback = client.create_feedback(st.session_state.run_id, "user_score", score=score)
            st.session_state.feedback = {"feedback_id": str(feedback.id), "score": score}
        tweet = generate_tweet_normal(topic)
        col4.markdown("### Prompted Tweet:")
        col4.write(f"🐦: {tweet}")
        col4.markdown("---")  # Add a horizontal line for separation
    else:
        st.warning("Please enter a topic before generating a tweet.")
