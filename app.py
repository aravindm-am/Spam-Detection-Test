import streamlit as st
import requests
import json

API_BASE = "http://163.69.82.203:8095/tmf/v1"

st.title("QoT Record API Interface")

mode = st.selectbox("Select Operation", ["Insert/Update", "Read/Query"])

msisdn = st.text_input("MSISDN (Phone Number)", max_chars=15)

if mode == "Insert/Update":
    st.subheader("Insert or Update QoT Record")

    src_o = st.text_input("Source Operator", "Jio")
    src_c = st.text_input("Source Country", "India")
    rep_o = st.text_input("Reported Operator", "Airtel")
    rep_c = st.text_input("Reported Country", "India")
    score = st.number_input("Score", min_value=0.0, max_value=1.0, value=0.1432, step=0.01)

    if st.button("Submit"):
        payload = {
            "requestId": "000001",
            "module": "tmforum",
            "channelID": "globalspamdatachannel",
            "chaincodeID": "qotcc",
            "functionName": "addQoTRecord",
            "payload": {
                "msisdn": msisdn,
                "src_o": src_o,
                "src_c": src_c,
                "rep_o": rep_o,
                "rep_c": rep_c,
                "score": score
            }
        }

        try:
            response = requests.post(f"{API_BASE}/invoke/", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            st.code(response.text, language="json")
        except Exception as e:
            st.error(f"Error: {e}")

elif mode == "Read/Query":
    st.subheader("Read QoT Record")

    if st.button("Fetch Record"):
        payload = {
            "requestId": "000001",
            "module": "tmforum",
            "channelID": "globalspamdatachannel",
            "chaincodeID": "qotcc",
            "functionName": "getQoTRecord",
            "payload": [msisdn]
        }

        try:
            response = requests.post(f"{API_BASE}/query/", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            st.code(response.text, language="json")
        except Exception as e:
            st.error(f"Error: {e}")
