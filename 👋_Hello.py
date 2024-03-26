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

st.set_page_config(page_title="Hello!", page_icon="👋")
st.write("# You made it to the ToxicityMod Demo! 👋")

st.sidebar.success("Check out different users")

st.markdown(
    """
    ToxicityMod can be used as a moderator bot for online communities. \n
    **👈 Select User or Owner view in the sidebar** to test out the model!

    ### User Mode
    - Send out any rude/profane or nice/happy messages to see how the Bot reacts
    - Keep sending messages until you get kicked!

    ### Owner Mode
    - Owners of the server can see stats on toxicity in their communities
    - Bot will flag concerning individuals for the Owner to review
"""
)


