import streamlit as st
from openai import OpenAI
from supabase import create_client

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model_id = "ft:gpt-4.1-mini-2025-04-14:pharmaai:test12:Biz0gbXo"

@st.cache_resource
def init_connection():
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

def validate_sql(sql):
    return sql.strip().upper().startswith("SELECT")

def log_feedback(original_question, clarified_question, interpretation, generated_sql, output_correct, feedback):
    supabase.table("feedback").insert({
        "original_question": original_question,
        "clarified_question": clarified_question,
        "ai_interpretation": interpretation,
        "generated_sql": generated_sql,
        "output_correct": output_correct,
        "feedback": feedback
    }).execute()

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

st.title("LLM PILOT")

for key in ["messages", "phase", "original_query", "clarified_query", "last_sql", "last_interpretation", "feedback_given", "additional_feedback"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "query" in key or "interpretation" in key else []

st.session_state.phase = st.session_state.phase or "waiting"

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
                        except Exception:
                            st.error("Query execution failed.")
                            log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                                         st.session_state.last_interpretation, sql, "Execution Error", "")
                            st.session_state.phase = "waiting"
                            st.stop()
                    else:
                        st.error("Only SELECT queries are allowed.")
                        log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                                     st.session_state.last_interpretation, sql, "Non-SELECT", "")
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
                st.markdown("Would you like to provide any additional feedback? (Yes/No)")
            st.session_state.phase = "ask_additional_feedback"

        elif user_input.lower() == "no":
            log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                         st.session_state.last_interpretation, st.session_state.last_sql, "No", "")
            with st.chat_message("assistant"):
                st.markdown("Okay, please clarify your question again.")
            st.session_state.phase = "clarify_intent"

    elif st.session_state.phase == "ask_additional_feedback":
        if user_input.lower() == "yes":
            with st.chat_message("assistant"):
                st.markdown("Please type your additional feedback.")
            st.session_state.phase = "collect_additional_feedback"
        elif user_input.lower() == "no":
            log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                         st.session_state.last_interpretation, st.session_state.last_sql, "Yes", "No additional feedback")
            with st.chat_message("assistant"):
                st.markdown("Thank you! Ask your next question anytime.")
            st.session_state.phase = "waiting"

    elif st.session_state.phase == "collect_additional_feedback":
        feedback = user_input
        log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                     st.session_state.last_interpretation, st.session_state.last_sql, "Yes", feedback)
        with st.chat_message("assistant"):
            st.markdown("Thanks for your feedback! You can ask another question.")
        st.session_state.phase = "waiting"
