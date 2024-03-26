
import streamlit as st
import pandas as pd
from streamlit_chat import message
from model import ToxicModel

st.set_page_config(page_title="User Mode", page_icon="🥊")
st.title("Welcome to 'Some Discord Server'!")
st.header("#rageroom")
st.text("Rage to your heart's content. \n The moderator will step in if it starts crossing the line.")

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if "toxicity_model" not in st.session_state:
    st.session_state["toxicity_model"] = ToxicModel(from_file="AaranyasBotModel.h5")
if "tracker" not in st.session_state:
    st.session_state["tracker"] = pd.Dataframe()

response_container = st.container()
container = st.container()

def response_generator(prompt):
    """ Edit to include the model outputs"""

    score = st.session_state["toxicity_model"].score(prompt)
    response = score["pretty"] + score["really"]
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # give warning and append to db
    # if third warning, kick from session, send response through system

    return response    


with container:
    # get the user input and add to the session
    user_input = st.chat_input("Go crazy") 
    #generate the response
    if user_input:
        output = response_generator(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# update the chat
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
