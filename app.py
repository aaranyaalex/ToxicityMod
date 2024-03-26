"""
Ideally we make this a discord bot

For portfolio purposes I will make it a streamlit app, that mimics the functionality

For the user:
- input their username and messages
- output responses from the bot accordingly:
    - warnings about toxicity, recommend to remove
    - warning about repeated flags
    - notification of removal
For the owner:
- input tolerance ranges (score to be notified, to be kicked from server)
- stats on toxic users via database, based on channels etc.

App should have:
1. Landing page for people to generate a username
2. two channels to input some messages for the bot
3. separate "Owner" tab to see stats and set parameters
"""

import streamlit as st
import random, time
from streamlit_chat import message
from model import ToxicModel
st.title("Welcome to 'Some Discord Server'!")

tab_titles = ["#rageroom","Server Owners Only!"]
tab1, tab2 = st.tabs(tab_titles)

# This is the mod simulator
with tab1:

    def response_generator(prompt):
        """ Edit to include the model outputs"""
        response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        st.session_state['messages'].append({"role": "user", "content": prompt})
        st.session_state['messages'].append({"role": "bot", "content": response})
        return response    
    
    
    
    st.header(tab_titles[0])
    st.subheader("Rage to your heart's content.")
    st.text("The moderator will step in if it starts crossing the line.")

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if "toxicity_model" not in st.session_state:
        st.session_state["toxicity_model"] = ToxicModel(from_file="aaranyasmodel.keras")

    response_container = st.container()
    container = st.container()

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


# this is the output visualizer
with tab2:
    st.header(tab_titles[1])
