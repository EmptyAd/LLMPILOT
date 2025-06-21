import streamlit as st
from openai import OpenAI
from supabase import create_client
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

client = OpenAI(api_key="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")
model_id = "ft:gpt-4.1-mini-2025-04-14:pharmaai:test12:Biz0gbXo"

@st.cache_resource
def init_connection():
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

def connect_to_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1Ec4WTZmAqR0r9kWWVmsf-UCSvy4RlIwvdSFBLUbpt8Y").worksheet("Feedback")
    return sheet

sheet = connect_to_gsheet()

def validate_sql(sql):
    return sql.strip().upper().startswith("SELECT")

schema_prompt = """You are a PostgreSQL expert helping to translate natural language questions into SQL queries.
You are working with a table named `patient_regimen_data` that contains the following columns and example values:

- vendor_patient_id (TEXT): Unique identifier for a patient, e.g. '34087654717'
- patient_age (INTEGER): Age of the patient, e.g. 73
- patient_gender (TEXT): Gender of the patient, e.g. 'M' or 'F'
- patient_state (TEXT): U.S. state code where the patient lives, e.g. 'CA'
- patient_status (TEXT): 'NEW' = indicates the patient is starting treatment for the condition for the first time, 'CONTINUE' = means the patient was already undergoing treatment for the condition before the current regimen began.
- is_active_flag (TEXT): Denotes if the patient is undergoing treatment as on the date of extraction of data. 'Y' = Active 'N' = Not Active
- line_of_therapy (TEXT): First-line therapy is the initial or primary treatment given.
Second-line (and beyond) refers to treatments given after the previous one has failed, stopped, or is no longer effective.
It helps track how many prior treatments the patient has received for the same condition.
Treatment sequence, e.g. '1', '4M', '5+' where M = maintenance
- start_date_of_therapy (TEXT): Start date of current regimen, e.g. '2022-01-15'
- end_date_of_therapy (TEXT): marks the date when the current treatment regimen was completed, discontinued, or altered in a way that ends the current line of therapy—such as stopping or adding one or more drugs from the regimen.
End date of current regimen, e.g. '2022-05-10'
- Client_regimen_group (TEXT): refers to the list or combination of drugs the patient is currently receiving as part of their ongoing treatment regimen. It represents the active set of medications in the current line of therapy., e.g. 'VELCADE + D'
- Current_Regimen_Length (INTEGER): Duration of current regimen in days, e.g. 120
- Previous_Regimen (TEXT): Refers to the list or combination of drugs the patient was on immediately before the current regimen, e.g. 'REVLIMID + DP'
- Previous_Regimen_Start (TEXT): Refers to the date when the previous treatment regimen was initiated for the patient , e.g. '2021-08-01'
- Previous_Regimen_End (TEXT): End date of previous regimen, e.g. '2021-12-30'
- Previous_Regimen_Length (INTEGER): Duration of previous regimen in days, e.g. 150
- Next_Regimen (TEXT): Refers to the list or combination of drugs the patient was on immediately after the current regimen, e.g. 'DARZALEX'
- Next_Regimen_Start (TEXT): Refers to the date when the next treatment regimen was initiated for the patient, e.g. '2023-01-01'
- Next_Regimen_End (TEXT): Date when the next regimen ended, e.g. '2023-06-01'
- Next_Regimen_Length (INTEGER): Duration in days of the patient's next (immediately after the current regimen) treatment regimen, e.g. 160

Only generate safe SELECT queries using these fields. Do not write INSERT, UPDATE, DELETE, or DROP statements."""

if "messages" not in st.session_state:
    st.session_state.messages = []
if "phase" not in st.session_state:
    st.session_state.phase = "waiting"
if "original_query" not in st.session_state:
    st.session_state.original_query = ""
if "clarified_query" not in st.session_state:
    st.session_state.clarified_query = ""
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_interpretation" not in st.session_state:
    st.session_state.last_interpretation = ""
if "followup_question" not in st.session_state:
    st.session_state.followup_question = ""
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False
if "additional_feedback" not in st.session_state:
    st.session_state.additional_feedback = ""

st.title("LLM PILOT")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask or clarify your SQL question")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.phase == "waiting":
        st.session_state.original_query = user_input
        st.session_state.feedback_given = False
        st.session_state.additional_feedback = ""
        with st.chat_message("assistant"):
            with st.spinner("Let me try to understand your question..."):
                interpretation = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You're a helpful assistant. Summarize what the user wants to know."},
                        {"role": "user", "content": user_input}
                    ]
                ).choices[0].message.content.strip()
            st.session_state.last_interpretation = interpretation
            msg = f"I understood your question as:\n\n**{interpretation}**\n\nPlease reply with Yes or No."
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.phase = "confirm_intent"

    elif st.session_state.phase == "confirm_intent":
        if user_input.lower() == "yes":
            query = st.session_state.clarified_query or st.session_state.original_query
            with st.chat_message("assistant"):
                with st.spinner("Generating SQL..."):
                    sql = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": schema_prompt},
                            {"role": "user", "content": query}
                        ]
                    ).choices[0].message.content.strip().rstrip(";")
                    st.session_state.last_sql = sql

                    if validate_sql(sql):
                        try:
                            result = supabase.rpc("run_raw_query", {"query": sql}).execute()
                            if result.data:
                                st.code(sql, language="sql")
                                st.dataframe(result.data)
                            else:
                                st.info("Query ran but returned no rows.")
                        except Exception as e:
                            st.error("Query execution failed.")
                            sheet.append_row([
                                st.session_state.original_query,
                                st.session_state.clarified_query,
                                st.session_state.last_interpretation,
                                sql,
                                "Execution Error",
                                ""  # Additional feedback column
                            ])
                            st.session_state.phase = "waiting"
                            st.stop()
                    else:
                        st.error("Only SELECT queries are allowed.")
                        sheet.append_row([
                            st.session_state.original_query,
                            st.session_state.clarified_query,
                            st.session_state.last_interpretation,
                            sql,
                            "Non-SELECT query blocked",
                            ""  # Additional feedback column
                        ])
                        st.session_state.phase = "waiting"
                        st.stop()

                followup = "Is this output correct? (Yes / No)"
                st.markdown(followup)
            st.session_state.messages.append({"role": "assistant", "content": f"Here is the SQL result.\n\n{followup}"})
            st.session_state.phase = "confirm_result"

        elif user_input.lower() == "no":
            with st.chat_message("assistant"):
                st.markdown("Please clarify what you meant.")
            st.session_state.phase = "clarify_intent"

        else:
            with st.chat_message("assistant"):
                st.markdown("Please reply with **Yes** or **No**.")

    elif st.session_state.phase == "clarify_intent":
        st.session_state.clarified_query = user_input
        with st.chat_message("assistant"):
            with st.spinner("Let me try to understand your clarification..."):
                interpretation = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You're a helpful assistant. Summarize what the user wants to know."},
                        {"role": "user", "content": user_input}
                    ]
                ).choices[0].message.content.strip()
            st.session_state.last_interpretation = interpretation
            msg = f"Got it. I understood your clarification as:\n\n**{interpretation}**\n\nShould I generate SQL for this? Yes / No"
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state.phase = "confirm_intent"

    elif st.session_state.phase == "confirm_result":
        if user_input.lower() == "yes":
            with st.chat_message("assistant"):
                st.markdown("Would you like to provide any additional feedback about this query or result? (Yes/No)")
            st.session_state.messages.append({"role": "assistant", "content": "Would you like to provide any additional feedback about this query or result? (Yes/No)"})
            st.session_state.phase = "ask_additional_feedback"

        elif user_input.lower() == "no":
            # Record the rejection
            sheet.append_row([
                st.session_state.original_query,
                st.session_state.clarified_query,
                st.session_state.last_interpretation,
                st.session_state.last_sql,
                "No",
                ""  # No additional feedback
            ])
            
            with st.chat_message("assistant"):
                st.markdown("Okay, please clarify your question again.")
            st.session_state.phase = "clarify_intent"

        else:
            with st.chat_message("assistant"):
                st.markdown("Please reply with Yes or No.")

    elif st.session_state.phase == "ask_additional_feedback":
        if user_input.lower() == "yes":
            with st.chat_message("assistant"):
                st.markdown("Please type your additional feedback about the query or results.")
            st.session_state.messages.append({"role": "assistant", "content": "Please type your additional feedback about the query or results."})
            st.session_state.phase = "collect_additional_feedback"
        elif user_input.lower() == "no":
            # Update the previous row with empty feedback
            sheet.append_row([
                st.session_state.original_query,
                st.session_state.clarified_query,
                st.session_state.last_interpretation,
                st.session_state.last_sql,
                "Yes",
                "No additional feedback"
            ])
            
            with st.chat_message("assistant"):
                st.markdown("Thank you! Ask your next question whenever you're ready.")
            st.session_state.messages.append({"role": "assistant", "content": "Thank you! Ask your next question whenever you're ready."})
            st.session_state.phase = "waiting"
        else:
            with st.chat_message("assistant"):
                st.markdown("Please reply with Yes or No.")

    elif st.session_state.phase == "collect_additional_feedback":
        st.session_state.additional_feedback = user_input
        # Update the previous row with the feedback
        sheet.append_row([
            st.session_state.original_query,
            st.session_state.clarified_query,
            st.session_state.last_interpretation,
            st.session_state.last_sql,
            "Yes",
            user_input
        ])
        
        with st.chat_message("assistant"):
            st.markdown("Thank you for your feedback!")
        st.session_state.messages.append({"role": "assistant", "content": "Thank you for your feedback!"})
        with st.chat_message("assistant"):
            st.markdown("Ask your next question whenever you're ready.")
        st.session_state.phase = "waiting"
