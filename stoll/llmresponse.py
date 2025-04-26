from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage

def load_chain(memory):
    template = """
    You are a note taker, you take in the user's audio and transcribe it into text. Also transcribe to a summrazied evalutaion of the conversation.

    The conversation transcript is as follows:
    {history}

    And here is the user's follow-up: {input}

    Your response:
    """
    # ---- LangChain LLM Setup ---- #
    llm = ChatOllama(model=MODEL, streaming=True)

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    chain = LLMChain(llm=llm, prompt=PROMPT, memory=memory)
    return chain


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response
