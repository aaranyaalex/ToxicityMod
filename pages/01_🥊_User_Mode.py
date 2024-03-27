
import streamlit as st
import pandas as pd
import random
from streamlit_chat import message
from model import ToxicModel

st.set_page_config(page_title="User Mode", page_icon="🥊")
st.title("Welcome to 'Some Discord Server'!")
st.header("#rageroom")
st.text("Rage to your heart's content. \nThe moderator will step in if it starts crossing the line.")

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
    st.session_state["tracker"] = pd.read_csv('demohistory.csv')
if "user" not in st.session_state:
    num = len(st.session_state["tracker"])
    st.session_state["user"] = "user_" + str(num +1)
    st.session_state["tracker"].loc[num] = ["Good", st.session_state["user"], 0, 0, 0, 0, 0, 0, 0]
if "kicked" not in st.session_state:
    st.session_state["kicked"] = False

response_container = st.container()
container = st.container()

def response_generator(prompt):
    """ Edit to include the model outputs"""

    raw, score = st.session_state["toxicity_model"].score(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})
    #st.session_state['messages'].append({"role": "assistant", "content": response})

    # do we need to issue a warning
    warn = 1 if score["score"] > 0.4 else 0
    kick = 0
    # add to the history
    st.session_state["tracker"].iloc[-1, 2:] += raw[0].tolist() + [warn]
    # do we need to kick them
    if 2 >= st.session_state["tracker"].iloc[-1, -1] >=1:
        st.session_state["tracker"].iloc[-1, 0] = "Toxic"
    elif st.session_state["tracker"].iloc[-1, -1] > 3:
        st.session_state["tracker"].iloc[-1, 0] = "Severely Toxic"
        kick = 1
        warn = 0

    return score, warn, kick    

with container:
    # get the user input and add to the session
    user_input = st.chat_input("Type here", disabled= st.session_state["kicked"]) 
    #generate the response
    if user_input:
        score, warn, kick = response_generator(user_input)
        st.session_state['past'].append(user_input)
        if kick:
            st.session_state['generated'].append("We've given you enough warnings. You've been reported.")
            st.session_state["kicked"] = True
            st.session_state['tracker'].to_csv('demohistory.csv', index=False)
            st.rerun()

        elif warn:
            if len(score["really"]) >= len(score["pretty"]):
                st.session_state['generated'].append("Woah! That's a really toxic message. You won't have many chances left...")
            else:
                st.session_state['generated'].append("Hey, that's kinda toxic. Consider being a little nicer.")
        else:
            response = random.choice(
                [
                    "Preach!",
                    "Kinda vanilla.",
                    "Ehh, my mom has said worse.",
                    "Maybe you could seek professional help?",
                    "C'mon now - no one can actually hear you here!",
                ]
            )
            st.session_state['generated'].append(response)

# update the chat
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
