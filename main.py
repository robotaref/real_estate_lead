import json
import os
import sys
import datetime
import openai
import pandas as pd
import streamlit as st

from audio_recorder_streamlit import audio_recorder

client = openai.OpenAI(
    api_key=""
)
CLIENT_NAME_PROMPT = "Name of the client that visited the house. If they mention someone's name, in the conversation as  use that."
STATE_PROMPT = "State of the visit."
DATE_PROMPT = "Date of the visit."
UNIT_PROMPT = "Unit number of the house. If it's a house, use 'house'."
LOCATION_PROMPT = "Location of the house."
house_leads = json.loads(open("house_leads.json").read())
MAIN_PROMPT = f"You are a real estate agent who is updating the backend database with house leads. " \
              f"You have a function called `update_house_leads` that takes in two parameters: " \
              f"`house_name` and `client_name`. " \
              f"The function updates the backend database with the house leads. " \
              f"Whenever get a new message, extract location and unit number of the house,  name of the client name," \
              f" date of visit and state of the visit." \
              f"This is how you define state: " \
              f"If they just want to visit the house in future, state will be " \
              f"'booked'. if they have visited the house, state will be 'done'. If they have cancelled the visit, " \
              f"state will be 'canceled'. Date should be in the format 'YYYY-MM-DD' and it should use real date. " \
              f"Update the backend database with this new lead." \
              f"If you can't interpret any of the required data, just mention it."


def transcribe(audio_file):
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",
    )
    return transcript


def save_audio_file(audio_bytes, file_extension):
    file_name = f"audio.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = transcribe(audio_file)
    print(transcript)
    return transcript.text


def update_backend(location, unit_number, date, client_name, state):
    added_row = {
        "location": location,
        "unit_number": unit_number,
        "date": date,
        "client_name": client_name,
        "state": state
    }
    house_leads.append(added_row)
    with open("house_leads.json", "w") as f:
        json.dump(house_leads, f)


def generate_response(message):
    prompt = (MAIN_PROMPT +
              f"current date time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "update_house_leads",
                    "description": "Updates backend database with house leads",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": LOCATION_PROMPT
                            },
                            "unit_number": {
                                "type": "string",
                                "description": UNIT_PROMPT
                            },
                            "date": {
                                "type": "string",
                                "description": DATE_PROMPT
                            },
                            "state": {
                                "type": "string",
                                "enum": ["done", "booked", "canceled"],
                                "description": STATE_PROMPT
                            },
                            "client_name": {
                                "type": "string",
                                "description": CLIENT_NAME_PROMPT
                            }
                        },
                        "required": ["location", "unit_number", "date", "state", "client_name"]
                    }
                }
            }, ],
    )
    if response.choices[0].message.tool_calls:
        for tool in response.choices[0].message.tool_calls:
            if tool.function.name == "update_house_leads":
                data = json.loads(tool.function.arguments)
                update_backend(**data)
                return data
    return response.choices[0].message.content


def main():
    st.title("Whisper Transcription")
    global CLIENT_NAME_PROMPT, STATE_PROMPT, DATE_PROMPT, UNIT_PROMPT, LOCATION_PROMPT, MAIN_PROMPT
    MAIN_PROMPT = st.sidebar.text_area("Main Prompt", MAIN_PROMPT)
    CLIENT_NAME_PROMPT = st.sidebar.text_input("Client Name Prompt", CLIENT_NAME_PROMPT)
    STATE_PROMPT = st.sidebar.text_input("State Prompt", STATE_PROMPT)
    DATE_PROMPT = st.sidebar.text_input("Date Prompt", DATE_PROMPT)
    UNIT_PROMPT = st.sidebar.text_input("Unit Prompt", UNIT_PROMPT)
    LOCATION_PROMPT = st.sidebar.text_input("Location Prompt", LOCATION_PROMPT)

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes, "mp3")

    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])
        if audio_file:
            file_extension = audio_file.type.split('/')[1]
            save_audio_file(audio_file.read(), file_extension)

    if st.button("Transcribe"):
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )

        transcript_text = transcribe_audio(audio_file_path)

        st.header("Transcript")
        st.write(transcript_text)
        data = generate_response(transcript_text)
        st.header("Extracted Data")
        st.write(data)
        st.download_button("Download Transcript", transcript_text)

    df = pd.DataFrame(house_leads, columns=["location", "unit_number", "date", "client_name", "state"])
    st.sidebar.header("House Leads")
    print(df.head())
    st.sidebar.dataframe(df)


if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    main()
