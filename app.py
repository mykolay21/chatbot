
import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

MODEL = "gpt-4o"

client = AzureOpenAI(
    api_key         = os.environ["AZURE_OPENAI_API_KEY"],
    api_version     = "2025-04-01-preview",
    azure_endpoint  = os.environ["AZURE_OPENAI_ENDPOINT"],
    timeout=60  # ← FIX
)

# The title method is used to display a text with big bold font at the page:
st.title("DIAL API Demo")

# The caption method is used to display a small text below the title:
st.caption("Streamlit application using DIAL API demo")

# Initialize chat history. Every time the app is reloaded, the chat history will be saved in the session state:
# st.session_state is a dictionary-like object that allows you to store information across multiple runs of the app.
# At the first run there is no "messages" key in the session state, so we initialize it with a system prompt:
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Display chat messages from history on app rerun. Every time the app is reloaded, the chat history will be displayed.
# st.chat_message is a container that can be used to display chat messages in a chat-like interface.
# The role parameter can be "user" or "assistant" to differentiate between user and assistant messages.
for message in st.session_state.messages:
    if message["role"] == "system":
        continue  # Skip displaying system messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input. The first time the app is run, this will be empty.
# When the user enters a message, the app will be reloaded and the message will be processed.
# st.chat_input is a text input box that can be used to accept user input in a chat-like interface.
# When the user submits a message, the prompt variable will contain the message text.
prompt = st.chat_input("How can I help you?")
if prompt:
    # Add user message to chat history. If don't add it, it will be lost when the app is reloaded.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


    # Calling the LLM API to get a response. We use the same messages for UI and for LLM, but we can create a separate list if needed.
    response = client.chat.completions.create(
        model=MODEL,
        messages=st.session_state.messages,
        timeout=60  # ← FIX
    )
    assistant_replied = (response.choices[0].message.content)
    # The response from the LLM will is added to the chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_replied})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_replied)
