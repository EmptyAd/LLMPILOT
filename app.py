import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from supabase import create_client
from rag_utils import retrieve_relevant_docs, load_docs_from_json

st.set_page_config(layout="wide", page_title="SQL Chat")

st.markdown("""
<style>
.element-container { margin-bottom: 0.5rem; }
/* Remove white background from chat input */
.stChatInput > div {
    background: transparent !important;
    border: none !important;
}
.stChatInput {
    background: transparent !important;
}
.graph-button {
    margin: 5px;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ddd;
    background-color: #f0f2f6;
    text-align: center;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Load RAG documents
load_docs_from_json("rag_docs.json")

# Initialize clients
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model_id = "ft:gpt-4.1-2025-04-14:pharmaai:26june3:BmcUSw4q"
model_interpretation = "gpt-4o-2024-08-06"

@st.cache_resource
def init_supabase():
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Helper functions
def is_data_related(query):
    response = client.chat.completions.create(
        model=model_interpretation,
        messages=[
            {
                "role": "system",
                "content": "You are a triage assistant. Your task is to classify user questions as either:\n\n- 'data': if the user is asking about patients, regimens, drugs, therapy lines, or anything related to SQL or data analysis.\n- 'out_of_context': if it's small talk, jokes, personal questions, requests about the AI itself, or anything unrelated to healthcare SQL data.\n\nOnly respond with: 'data' or 'out_of_context'."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    ).choices[0].message.content.strip().lower()
    return response.startswith("data")

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

def suggest_graph_type(df):
    """Suggest appropriate graph types based on data characteristics"""
    if df is None or df.empty:
        return []
    
    suggestions = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Basic suggestions based on data structure
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        suggestions.extend(['Bar Chart'])
    
    if len(numeric_cols) >= 2:
        suggestions.extend(['Scatter Plot', 'Line Chart'])
    
    if len(categorical_cols) >= 1:
        suggestions.extend(['Pie Chart', 'Count Plot'])
    
    # Always include histogram for numeric data
    if len(numeric_cols) >= 1:
        suggestions.append('Histogram')
    
    # Remove duplicates and return
    return list(set(suggestions))

def create_graph(df, graph_type, x_col=None, y_col=None, color_col=None):
    """Create different types of graphs based on user selection"""
    if df is None or df.empty:
        return None
    
    try:
        if graph_type == "Bar Chart":
            if x_col and y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                           title=f"Bar Chart: {y_col} by {x_col}")
            else:
                # Default: use first categorical and first numeric
                cat_cols = df.select_dtypes(include=['object', 'string']).columns
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(cat_cols) > 0 and len(num_cols) > 0:
                    fig = px.bar(df, x=cat_cols[0], y=num_cols[0], 
                               title=f"Bar Chart: {num_cols[0]} by {cat_cols[0]}")
                else:
                    return None
        
        elif graph_type == "Pie Chart":
            if x_col:
                value_counts = df[x_col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Pie Chart: Distribution of {x_col}")
            else:
                # Use first categorical column
                cat_cols = df.select_dtypes(include=['object', 'string']).columns
                if len(cat_cols) > 0:
                    value_counts = df[cat_cols[0]].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                               title=f"Pie Chart: Distribution of {cat_cols[0]}")
                else:
                    return None
        
        elif graph_type == "Line Chart":
            if x_col and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col,
                            title=f"Line Chart: {y_col} vs {x_col}")
            else:
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(num_cols) >= 2:
                    fig = px.line(df, x=num_cols[0], y=num_cols[1],
                                title=f"Line Chart: {num_cols[1]} vs {num_cols[0]}")
                else:
                    return None
        
        elif graph_type == "Scatter Plot":
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"Scatter Plot: {y_col} vs {x_col}")
            else:
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(num_cols) >= 2:
                    fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                                   title=f"Scatter Plot: {num_cols[1]} vs {num_cols[0]}")
                else:
                    return None
        
        elif graph_type == "Histogram":
            if x_col:
                fig = px.histogram(df, x=x_col, title=f"Histogram: {x_col}")
            else:
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(num_cols) > 0:
                    fig = px.histogram(df, x=num_cols[0], title=f"Histogram: {num_cols[0]}")
                else:
                    return None
        
        elif graph_type == "Box Plot":
            if x_col and y_col:
                fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot: {y_col} by {x_col}")
            else:
                cat_cols = df.select_dtypes(include=['object', 'string']).columns
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(cat_cols) > 0 and len(num_cols) > 0:
                    fig = px.box(df, x=cat_cols[0], y=num_cols[0],
                               title=f"Box Plot: {num_cols[0]} by {cat_cols[0]}")
                else:
                    return None
        
        elif graph_type == "Count Plot":
            if x_col:
                value_counts = df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Count Plot: {x_col}")
            else:
                cat_cols = df.select_dtypes(include=['object', 'string']).columns
                if len(cat_cols) > 0:
                    value_counts = df[cat_cols[0]].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Count Plot: {cat_cols[0]}")
                else:
                    return None
        
        else:
            return None
        
        # Update layout for better appearance
        fig.update_layout(
            height=400,
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating graph: {str(e)}")
        return None

schema_prompt = """You are a PostgreSQL expert helping to translate natural language questions into PSQL queries.
You are working with a table named `patient_regimen_data` that contains the following columns and example values:

- patient_id (TEXT): Unique identifier for a patient, e.g. '34087654717'
- patient_age (INTEGER): Age of the patient, e.g. 73
- patient_gender (TEXT): Gender of the patient, e.g. 'M' or 'F'
- patient_state (TEXT): U.S. state code where the patient lives, e.g. 'CA'
- patient_status (TEXT): 'NEW' = indicates the patient is starting treatment for the condition for the first time, 'CONTINUE' = means the patient was already undergoing treatment for the condition before the current regimen began.
- is_active_flag (TEXT): Denotes if the patient is undergoing treatment as on the date of extraction of data. 'Y' = Active 'N' = Not Active
- line_of_therapy (TEXT): First-line therapy is the initial or primary treatment given.
Second-line (and beyond) refers to treatments given after the previous one has failed, stopped, or is no longer effective.
It helps track how many prior treatments the patient has received for the same condition.
Treatment sequence, e.g. '1', '4M', '5+' where M = maintenance
- Current_Regimen_Start (TEXT): Start date of current regimen, e.g. '2022-01-15'
- Current_Regimen_End (TEXT): marks the date when the current treatment regimen was completed, discontinued, or altered in a way that ends the current line of therapy—such as stopping or adding one or more drugs from the regimen.
End date of current regimen, e.g. '2022-05-10'
- Current_Regimen (TEXT): refers to the list or combination of drugs the patient is currently receiving as part of their ongoing treatment regimen. It represents the active set of medications in the current line of therapy., e.g. 'VELCADE + D'
- Current_Regimen_Length (INTEGER): Duration of current regimen in days, e.g. 120
- Previous_Regimen (TEXT): Refers to the list or combination of drugs the patient was on immediately before the current regimen, e.g. 'REVLIMID + DP'
- Previous_Regimen_Start (TEXT): Refers to the date when the previous treatment regimen was initiated for the patient , e.g. '2021-08-01'
- Previous_Regimen_End (TEXT): End date of previous regimen, e.g. '2021-12-30'
- Previous_Regimen_Length (INTEGER): Duration of previous regimen in days, e.g. 150
- Next_Regimen (TEXT): Refers to the list or combination of drugs the patient was on immediately after the current regimen, e.g. 'DARZALEX'
- Next_Regimen_Start (TEXT): Refers to the date when the next treatment regimen was initiated for the patient, e.g. '2023-01-01'
- Next_Regimen_End (TEXT): Date when the next regimen ended, e.g. '2023-06-01'
- Next_Regimen_Length (INTEGER): Duration in days of the patient's next (immediately after the current regimen) treatment regimen, e.g. 160

Note:
-All the names of drug or the medicines are written in all capital e.g. DOXIL, VELCADE, DARZALEX, etc.
Data Model:
- Each row represents a regimen for a patient.
- Each drug in a regimen is stored as a comma-separated string in the Current_Regimen, Previous_Regimen, and Next_Regimen fields.

Temporal Logic
- Add-On Definition: A drug is considered an add-on when it appears in the Next_Regimen but was not in the Current_Regimen, and the Current_Regimen drugs are still present in the next regimen.
- Switch Definition: A switch is when the drugs in Current_Regimen are replaced by entirely different drugs in Next_Regimen and is not present in the Next_Regimen.

The data you will query is based on patient-level transactions transformed into regimen-level data. A regimen represents the set of drugs a patient is on at a particular point in time. Your PSQL should reflect precise temporal and treatment logic as described above.
Only generate safe SELECT queries using these fields. Do not write INSERT, UPDATE, DELETE, or DROP statements."""

# Initialize session state
for key in ["messages", "phase", "original_query", "clarified_query", "last_sql", "last_interpretation", "feedback_given", "additional_feedback", "last_error", "sql_result", "current_graph", "show_graph_options"]:
    if key not in st.session_state:
        if key in ["messages", "feedback_given"]:
            st.session_state[key] = []
        elif key in ["original_query", "clarified_query", "last_sql", "last_interpretation", "last_error"]:
            st.session_state[key] = ""
        elif key in ["show_graph_options"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

st.session_state.phase = st.session_state.phase or "waiting"

# Main layout
left_col, right_col = st.columns([1, 2])

# LEFT: Chat
with left_col:
    st.markdown("### 💬 Chat")
    messages_container = st.container(height=500)
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    user_input = st.chat_input("Ask your SQL question")
    
    if user_input:
        # Add user message to session state and rerun to display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Phase logic from your original code
        if st.session_state.phase == "waiting":
            st.session_state.original_query = user_input
            st.session_state.clarified_query = ""
            st.session_state.last_sql = ""
            st.session_state.last_error = ""
            st.session_state.sql_result = None
            st.session_state.current_graph = None
            st.session_state.show_graph_options = False
            
            with st.spinner("Let me try to understand your question..."):
                interpretation = client.chat.completions.create(
                    model=model_interpretation,
                    messages=[
                        {"role": "system", "content":"""You're a helpful assistant and you need to summarize what the user wants to know, 
                                            such that when another AI recieves the output of your understanding, it could generate a psql query to extract the data.
                                            Dont go overboard and get into the details of the drugs and how or why is it used, stick to just the understanding of the question.
                                            Note: 
                                            - All the names of drug or the medicines are written in all capital e.g. DOXIL, VELCADE, DARZALEX, etc.
                                            - Regimen: A combination of drugs a patient is on during a specific treatment period.
                                            - Add-On Definition: A drug is considered an add-on when it appears in the Next_Regimen but was not in the Current_Regimen, and the Current_Regimen drugs are still present in the next regimen.
                                            - Switch Definition: A switch is when the drugs in Current_Regimen are replaced by entirely different drugs in Next_Regimen and is not present in the Next_Regimen.
                                            - SOB (Source of Business): Refers to the previous drug(s) or regimen(s) a patient was on before transitioning to a new one.
                                            - LOT (Line of Therapy or Length of Therapy):
                                              - Line of Therapy: Indicates the sequence in which treatments were administered (e.g., 1st-line, 2nd-line).
                                              - Length of Therapy: Refers to the duration a patient stayed on a regimen, typically calculated as Current_Regimen_End - Current_Regimen_Start.
                                            """},  
                        {"role": "user", "content": user_input}
                    ]
                ).choices[0].message.content.strip()
            st.session_state.last_interpretation = interpretation
            msg = f"I understood your question as:\n\n**{interpretation}**\n\nPlease reply with Yes or No."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.phase = "confirm_intent"

        elif st.session_state.phase == "confirm_intent":
            if user_input.lower() == "yes":
                query = st.session_state.clarified_query or st.session_state.original_query
                docs = retrieve_relevant_docs(query)
                context = "\n\n".join(docs)
                sql_prompt = f"""You are a PostgreSQL expert helping translate user questions into SQL queries. Use the context below and follow all business logic carefully.

                                ### Context:
                                {context}

                                ### User Question:
                                "{query}"

                                ### AI Interpretation:
                                "{st.session_state.last_interpretation}"

                                {f'### Previous SQL Error:{st.session_state.last_error}' if st.session_state.last_error else ''}

                                ### Previous SQL (if any):
                                {st.session_state.last_sql or '[None]'}

                                Write a safe, syntactically correct SELECT query. 
                                Strictly apply definitions of 'add-on', 'switch', regimen timelines, and patient status.
                                Avoid guessing if data is missing. Only use fields described in the schema."""

                with st.spinner("Generating SQL..."):
                    sql = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": schema_prompt},
                            {"role": "user", "content": sql_prompt}
                        ]
                    ).choices[0].message.content.strip().rstrip(";")

                    st.session_state.last_sql = sql

                    if validate_sql(sql):
                        try:
                            result = supabase.rpc("run_raw_query", {"query": sql}).execute()
                            if result.data:
                                st.session_state.sql_result = result.data
                                st.session_state.show_graph_options = True
                                st.session_state.phase = "confirm_result"
                                followup = "Generated SQL and executed it successfully!\n\nIs this output correct? Yes / No"
                                st.session_state.messages.append({"role": "assistant", "content": followup})
                            else:
                                st.session_state.phase = "confirm_result"
                                no_rows_msg = "Query executed but returned no rows. Is this correct? Yes / No"
                                st.session_state.messages.append({"role": "assistant", "content": no_rows_msg})
                        except Exception as e:
                            error_msg = str(e)
                            st.session_state.last_error = error_msg
                            st.session_state.phase = "retry_on_error"
                            retry_msg = f"SQL execution failed: {error_msg[:100]}... Query failed. Want me to try again with this error? Yes / No"
                            st.session_state.messages.append({"role": "assistant", "content": retry_msg})
                    else:
                        log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                                    st.session_state.last_interpretation, sql, "Non-SELECT", "")
                        st.session_state.phase = "waiting"
                        st.session_state.messages.append({"role": "assistant", "content": "Invalid query type. Only SELECT queries allowed. Please ask a new question."})

            elif user_input.lower() == "no":
                st.session_state.messages.append({"role": "assistant", "content": "Please clarify what you meant."})
                st.session_state.phase = "clarify_intent"

        elif st.session_state.phase == "retry_on_error":
            if user_input.lower() == "yes":
                st.session_state.phase = "confirm_intent"
                st.session_state.messages.append({"role": "assistant", "content": "Let me try again with the error information..."})
                # Trigger rerun to process confirm_intent phase
                st.rerun()
            elif user_input.lower() == "no":
                log_feedback(
                    st.session_state.original_query,
                    st.session_state.clarified_query,
                    st.session_state.last_interpretation,
                    st.session_state.last_sql,
                    "No",
                    st.session_state.last_error
                )
                st.session_state.messages.append({"role": "assistant", "content": "Thanks! Would you like to give feedback? Yes/No"})
                st.session_state.phase = "ask_additional_feedback"

        elif st.session_state.phase == "clarify_intent":
            st.session_state.clarified_query = user_input
            with st.spinner("Let me try to understand your clarification..."):
                clarification_prompt = f"""The original question was:
{st.session_state.original_query}
The AI understood the original question as:
{st.session_state.last_interpretation}
The user clarified it as:
{user_input}
Please summarize what the user actually wants now."""
                interpretation = client.chat.completions.create(
                    model=model_interpretation,
                    messages=[
                        {"role": "system", "content": """You're a helpful assistant and you need to summarize what the user wants to know, 
                                            such that when another AI recieves the output of your understanding, it could generate a psql query to extract the data.
                                            Dont go overboard and get into the details of the drugs and how or why is it used, stick to just the understanding of the question.
                                            Note: 
                                            - All the names of drug or the medicines are written in all capital e.g. DOXIL, VELCADE, DARZALEX, etc.
                                            - Regimen: A combination of drugs a patient is on during a specific treatment period.
                                            - Add-On Definition: A drug is considered an add-on when it appears in the Next_Regimen but was not in the Current_Regimen, and the Current_Regimen drugs are still present in the next regimen.
                                            - Switch Definition: A switch is when the drugs in Current_Regimen are replaced by entirely different drugs in Next_Regimen and is not present in the Next_Regimen.
                                            - SOB (Source of Business): Refers to the previous drug(s) or regimen(s) a patient was on before transitioning to a new one.
                                            - LOT (Line of Therapy or Length of Therapy):
                                              - Line of Therapy: Indicates the sequence in which treatments were administered (e.g., 1st-line, 2nd-line).
                                              - Length of Therapy: Refers to the duration a patient stayed on a regimen, typically calculated as Current_Regimen_End - Current_Regimen_Start.
                                            """},
                        {"role": "user", "content": clarification_prompt}
                    ]
                ).choices[0].message.content.strip()

            st.session_state.last_interpretation = interpretation
            msg = f"Got it. I understood your clarification as:\n\n**{interpretation}**\n\nShould I generate SQL for this? Yes / No"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.phase = "confirm_intent"

        elif st.session_state.phase == "confirm_result":
            if user_input.lower() == "yes":
                st.session_state.messages.append({"role": "assistant", "content": "Would you like to provide any additional feedback? Yes/No"})
                st.session_state.phase = "ask_additional_feedback"
            elif user_input.lower() == "no":
                log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                             st.session_state.last_interpretation, st.session_state.last_sql, "No", "")
                st.session_state.messages.append({"role": "assistant", "content": "Okay, please clarify your question again."})
                st.session_state.phase = "clarify_intent"

        elif st.session_state.phase == "ask_additional_feedback":
            if user_input.lower() == "yes":
                st.session_state.messages.append({"role": "assistant", "content": "Please type your additional feedback."})
                st.session_state.phase = "collect_additional_feedback"
            elif user_input.lower() == "no":
                log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                             st.session_state.last_interpretation, st.session_state.last_sql, "Yes", "No additional feedback")
                st.session_state.messages.append({"role": "assistant", "content": "Thank you! Ask your next question anytime."})
                st.session_state.phase = "waiting"

        elif st.session_state.phase == "collect_additional_feedback":
            feedback = user_input
            log_feedback(st.session_state.original_query, st.session_state.clarified_query,
                         st.session_state.last_interpretation, st.session_state.last_sql, "Yes", feedback)
            st.session_state.messages.append({"role": "assistant", "content": "Thanks for your feedback! You can ask another question."})
            st.session_state.phase = "waiting"
        
        st.rerun()

# RIGHT: Output and Graphs
with right_col:
    st.title("📊 SQL Output & Visualizations")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📋 Data Results", "📈 Graphs"])
    
    with tab1:
        if st.session_state.last_sql:
            st.subheader("Generated SQL")
            st.code(st.session_state.last_sql, language="sql")
        
        if st.session_state.sql_result:
            st.subheader("Query Results")
            df = pd.DataFrame(st.session_state.sql_result)
            st.dataframe(df, use_container_width=True)
            
            # Show data summary
            if not df.empty:
                with st.expander("📊 Data Summary"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", len(df))
                        st.metric("Total Columns", len(df.columns))
                    with col2:
                        numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
                        categorical_cols = len(df.select_dtypes(include=['object', 'string']).columns)
                        st.metric("Numeric Columns", numeric_cols)
                        st.metric("Categorical Columns", categorical_cols)
    
    with tab2:
        if st.session_state.sql_result and st.session_state.show_graph_options:
            df = pd.DataFrame(st.session_state.sql_result)
            
            if not df.empty:
                st.subheader("🎨 Create Visualizations")
                
                # Get column information
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                all_cols = df.columns.tolist()
                
                # Graph type selection
                suggested_graphs = suggest_graph_type(df)
                
                if suggested_graphs:
                    st.write("**Suggested graph types based on your data:**")
                    
                    # Create buttons for each suggested graph type
                    cols = st.columns(min(3, len(suggested_graphs)))
                    for i, graph_type in enumerate(suggested_graphs):
                        with cols[i % 3]:
                            if st.button(f"📊 {graph_type}", key=f"btn_{graph_type}"):
                                fig = create_graph(df, graph_type)
                                if fig:
                                    st.session_state.current_graph = fig
                                    st.session_state.current_graph_type = graph_type
                
                # Advanced graph options
                with st.expander("🔧 Advanced Graph Options"):
                    st.write("**Customize your visualization:**")
                    
                    graph_type = st.selectbox(
                        "Select Graph Type",
                        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", 
                         "Histogram", "Box Plot", "Count Plot"],
                        key="custom_graph_type"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        x_col = st.selectbox("X-axis", [None] + all_cols, key="x_col")
                    
                    with col2:
                        y_col = st.selectbox("Y-axis", [None] + numeric_cols, key="y_col")
                    
                    with col3:
                        color_col = st.selectbox("Color by", [None] + categorical_cols, key="color_col")
                    
                    if st.button("🎨 Create Custom Graph", key="create_custom"):
                        fig = create_graph(df, graph_type, x_col, y_col, color_col)
                        if fig:
                            st.session_state.current_graph = fig
                            st.session_state.current_graph_type = graph_type
                
                # Display the current graph
                if st.session_state.current_graph:
                    st.subheader(f"📈 {st.session_state.get('current_graph_type', 'Visualization')}")
                    st.plotly_chart(st.session_state.current_graph, use_container_width=True)
                    
                    # Download options
                    col1, col2 = st.columns(2)x
                    with col1:
                        if st.button("💾 Download as PNG"):
                            st.session_state.current_graph.write_image("chart.png")
                            st.success("Chart saved as chart.png")
                    
                    with col2:
                        if st.button("📋 Download as HTML"):
                            st.session_state.current_graph.write_html("chart.html")
                            st.success("Chart saved as chart.html")
            
            else:
                st.info("No data available to create graphs. Please run a SQL query first.")
        
        elif not st.session_state.show_graph_options:
            st.info("🎯 Run a SQL query first to see visualization options!")
        
        else:
            st.info("📊 No data available for visualization.")
