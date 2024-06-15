# tweet-generator
LLM tweet generator


## Run a first time
pip install langchain-cli
langchain-cli --version # <-- Make sure the version is at least 0.0.22
## Will replace from langchain.chat_models import ChatOpenAI
langchain-cli migrate --diff train_model.py # Preview
langchain-cli migrate train_model.py # Apply