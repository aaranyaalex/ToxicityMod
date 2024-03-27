import streamlit as st
import pandas as pd


st.set_page_config(page_title="Owner Mode", page_icon="🧠")
container  = st.container()

if "tracker" not in st.session_state:
    st.session_state["tracker"] = pd.read_csv('demohistory.csv')
if "user" not in st.session_state:
    num = len(st.session_state["tracker"])
    st.session_state["user"] = "user_" + str(num +1)
    st.session_state["tracker"].loc[num] = ["Good", st.session_state["user"], 0, 0, 0, 0, 0, 0, 0]
if "kicked" not in st.session_state:
    st.session_state["kicked"] = False

with container:
    df = st.session_state["tracker"]
    df = df.set_index(df.columns[0])
    ratings = st.multiselect(
        "Filter by rating", ["Toxic", "Good", "Severely Toxic"], ["Toxic", "Severely Toxic"]
    )
    if not ratings:
        st.error("Please select at least one.")
    else:
        data = df.loc[ratings]
        #data /= 1000000.0
        st.write("### Toxic Users", data.sort_index())

    st.write("### Recent Activity")
    if st.session_state["kicked"]:
        st.write("- Just banned ", st.session_state["user"], "!")