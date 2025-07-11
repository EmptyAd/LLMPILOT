# NEW START:
schema_prompt = """You are a PostgreSQL expert helping to translate natural language questions into PSQL queries.
You are working with a table named `patient_regimen_data` and `prescriber_data` that contains the following columns and example values:

patient_regimen_data:
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
- initiated_drugs (TEXT): drugs which are present in next_regimen but not in current_regimen 
- discontinued_drugs (TEXT): drugs which are present in current_regimen but not next_regimen
- common_drugs (TEXT): drug which is present in both the current_regimen and next_regimen 
- prescriber_id (INTEGER): Unique numeric identifier for each prescriber, foreign key refering to the table prescriber_data.

prescriber_data:
- prescriber_id (INTEGER): Unique numeric identifier for each prescriber in the dataset
- specialty (TEXT): Medical specialty of the prescriber. Common values include:
med onc = Medical Oncologist
rad onc = Radiation Oncologist
NP/PA = Nurse Practitioner or Physician Assistant
others = Other specialties
- setting (TEXT): The healthcare environment in which the prescriber practices. Typical values:
academic = Academic medical center or teaching hospital
community = Community hospital or non-academic setting
- kol (TEXT): Indicates if the prescriber is a Key Opinion Leader.
yes = Recognized KOL
no = Not a KOL
- quintile (INTEGER): Influence or performance ranking of the prescriber on a 0–5 scale. 5 being the top 20% quintile and 0 being the bottom 20% quintile. This is based on the volume of prescriptions
- segment (TEXT): Behavioral or marketing segment assigned to the prescriber based on treatment style and engagement.
- state (TEXT): U.S. state abbreviation indicating where the prescriber practices (e.g., CA, OH, NJ).


Note:
-All the names of drug or the medicines are written in all capital e.g. DOXIL, VELCADE, DARZALEX, etc.
Data Model:
- Each row represents a regimen for a patient.
- Each drug in a regimen is stored as a comma-separated string in the Current_Regimen, Previous_Regimen, and Next_Regimen fields.

Temporal Logic
- Add-On Definition: A drug is considered an add-on when it appears in the Next_Regimen but was not in the Current_Regimen, and the Current_Regimen drugs are still present in the next regimen.
- Switch Definition: A switch is when the drugs in Current_Regimen are replaced by entirely different drugs in Next_Regimen and is not present in the Next_Regimen.
- Lookback Period: the duration prior to the start date of the current regimen used to determine whether a patient is new to a specific drug or the market (indication).
- New to Drug: If the patient has not received the given drug during the lookback period, they are considered new to that drug.
- New to Market: If the patient has not received any drug for the same indication during the lookback period, they are considered new to the market.
- Lookforward Period: the duration after the end date of the current regimen used to determine whether a patient has discontinued a specific drug or exited the treatment market.
- Drug Discontinued: If the patient does not receive the same drug in the lookforward period, they are considered to have discontinued that drug.
- Market Discontinued: If the patient does not receive any drug for the same indication in the lookforward period, they are considered to have discontinued from the market.

The data you will query is based on patient-level transactions transformed into regimen-level data. A regimen represents the set of drugs a patient is on at a particular point in time. Your PSQL should reflect precise temporal and treatment logic as described above.
Only generate safe SELECT queries using these fields. Do not write INSERT, UPDATE, DELETE, or DROP statements."""



interpretation_prompt ="""You're a helpful assistant and you need to summarize what the the question means, 
                            such that when another AI recieves the output of your understanding, it could generate a psql query to extract the data.
                            Be polite since your output will also be recieved by the user so it needs to be written in such a way user also understands it.
                            Dont go overboard and get into the details of the drugs and how or why is it used, stick to just the understanding of the question.
                                Note: 
                                    - All the names of drug or the medicines are written in all capital e.g. DOXIL, VELCADE, DARZALEX, etc.
                                    - Regimen: A combination of drugs a patient is on during a specific treatment period.
                                    - Add-On Definition: A drug is considered an add-on when it appears in the Next_Regimen but was not in the Current_Regimen, and the Current_Regimen drugs are still present in the next regimen.
                                    - Switch Definition: A switch is when the drugs in Current_Regimen are replaced by entirely different drugs in Next_Regimen and is not present in the Next_Regimen.
                                    - SOB (Source of Business): Refers to the previous drug(s) or regimen(s) a patient was on before transitioning to a new one.
                                    - LOT (Line of Therapy or Length of Therapy):
                                      - Line of Therapy: Indicates the sequence in which treatments were administered (e.g., 1st-line, 2nd-line).
                                      - Length of Therapy: Refers to the duration a patient stayed on a regimen, typically calculated as Current_Regimen_End - Current_Regimen_Start."""

medical_response_prompt = """You are a medical AI assistant with expertise in oncology and diagnostic imaging.


Instructions:
1. Carefully read the full CONTEXT below.
2. Even if exact keywords aren't present.
3. Attempt to answer the QUESTION using any direct or indirect information.
4. Use quotes, facts, or paraphrased insights from the context to support your answer.
5. If some relevant terms are present but the comparison is incomplete, provide a partial answer and indicate missing parts.
6. ONLY respond with "No relevant information found" if **absolutely nothing related** is found.
7. Give the answer in a very good and detailed format with bullets."""

classification_ques_prompt = """Analyze the following question and classify it into one of these categories:
    
    1. SQL - Questions about specific patient data, counts, filters, or requiring database queries
    2. RAG - Questions about definitions, explanations, or general knowledge related to breast cancer or any other medical terms
    3. OUT_OF_SCOPE - Unrelated questions
    
    Examples:
    - "How many patients received DARZALEX?" → SQL
    - "What role does HER2 overexpression play in the effectiveness of T-DM1?" → RAG
    - "Whats the weather today" → OUT_OF_SCOPE
    
    Return only the category name (SQL, RAG, or OUT_OF_SCOPE)."""
#NEW END

import streamlit as st
from openai import OpenAI
from supabase import create_client
import pandas as pd
from qdrant_client import QdrantClient
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model_id = "ft:gpt-4.1-2025-04-14:pharmaai:8july1:BqvJeIdp:ckpt-step-88"
model_interpretation = "gpt-4o-2024-08-06"

@st.cache_resource
def init_connection():
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# NEW START
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qdrant = QdrantClient(host="localhost", port=6333)
top_k = 5  
max_tokens = 4000  

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 1536

try:
    collections = qdrant.get_collections().collections
    if not collections:
        st.warning("❌ No Qdrant collections found. RAG functionality will be disabled.")
        collection_name = None
    else:
        collection_name = collections[0].name
        logger.info(f"✅ Using Qdrant collection: {collection_name}")
        info = qdrant.get_collection(collection_name)
except Exception as e:
    st.warning(f"❌ Failed to connect to Qdrant: {str(e)}")
    collection_name = None

def enhanced_rag_query(question: str) -> str:
    """Simplified and reliable RAG pipeline that always tries to answer using Qdrant + OpenAI."""
    if not collection_name:
        return "❌ RAG disabled (no Qdrant collection found)."

    try:
        embedding_response = client.embeddings.create(
            input=[preprocess_question(question)],
            model=EMBEDDING_MODEL
        )
        query_vector = embedding_response.data[0].embedding
        if len(query_vector) != 1536:
            logger.warning(f"⚠️ Embedding size mismatch: {len(query_vector)}")

        hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.1
        )
        if not hits:
            logger.info("🔁 No hits found with threshold, retrying without threshold.")
            hits = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k
            )

        if not hits:
            return f"❌ No relevant information found in the knowledge base for: '{question}'"

        context_chunks = []
        for i, hit in enumerate(hits):
            content = ""
            if hasattr(hit, "payload") and hit.payload:
                content = hit.payload.get("page_content", "").strip()
            if content:
                context_chunks.append(content)
                logger.info(f"✅ Source {i+1} added. Score: {hit.score:.3f}")
        
        if not context_chunks:
            return "❌ Retrieved documents had no usable 'page_content'."

        context = "\n\n".join(context_chunks)
        if len(context) > 12000:
            logger.warning("⚠️ Truncating context for token limit")
            context = context[:12000]

        logger.info(f"🧠 Context length: {len(context)} | Chunks: {len(context_chunks)}")

        response = client.chat.completions.create(
            model=model_interpretation,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the context below to answer the user's question accurately, even if only partial information is available."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.1,
            max_tokens=1500
        )

        if not response.choices:
            return "❌ No response generated."

        answer = response.choices[0].message.content.strip()

        if "no relevant information" in answer.lower() and context.strip():
            logger.warning("⚠️ Model returned 'no info' despite having context. Overriding.")
            return (
                "The system could not generate a full answer, but related information was available.\n\n"
                "Please try rephrasing the question or ask for clarification."
            )

        return answer

    except Exception as e:
        logger.error(f"❌ RAG query failed: {str(e)}", exc_info=True)
        return f"❌ Error: {str(e)}"

def preprocess_question(question: str) -> str:
    """Expands medical terms to improve semantic matching in vector search."""
    expansions = {
        "types of": "categories classifications varieties forms kinds",
        "breast cancer": "mammary carcinoma oncology tumor malignancy neoplasm",
        "treatment": "therapy medication drug regimen protocol",
        "drug": "medicine compound therapeutic agent",
        "mechanism": "mode of action pathway receptor",
        "side effects": "adverse events toxicity complications reactions",
        "dosage": "dose administration schedule",
        "efficacy": "effectiveness outcome response"
    }

    original = question
    question_lower = question.lower()

    for term, expansion in expansions.items():
        if term in question_lower:
            question += " " + expansion

    return f"{original} {question}"

def generate_medical_response(question: str, context: str, sources: List[Dict]) -> str:
    """Generate a medically grounded response based on retrieved context using OpenAI."""
    
    system_prompt_template = """{medical_response_prompt}

CONTEXT:
{context}
"""
    system_prompt = system_prompt_template.format(context=context)

    try:
        response = client.chat.completions.create(
            model=model_interpretation,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nCompare mammography and ultrasound or describe any differences or similarities between them based on the above context. Even if partial information is found, include it."}
            ],
            temperature=0.1,
            max_tokens=1500
        )

        if not response.choices:
            return "❌ No response generated by the model."

        answer = response.choices[0].message.content.strip()

        if "no relevant information" in answer.lower() and context.strip():
            logger.warning("⚠️ Model returned 'no info' despite context presence. Overriding.")
            answer = (
                "The system could not generate a full answer, but related information was available. "
                "Please rephrase your question for better results."
            )

        if sources:
            avg_score = sum(s["score"] for s in sources) / len(sources)
            avg_relevance = sum(s["relevance"] for s in sources) / len(sources)
            answer += f"\n\n🔍 **Source confidence**: {avg_score:.2f} | **Term relevance**: {avg_relevance:.1%}"

        return answer

    except Exception as e:
        logger.error(f"❌ OpenAI generation failed: {str(e)}", exc_info=True)
        return f"❌ Failed to generate answer. Error: {str(e)}"

def calculate_relevance(question: str, context: str) -> float:
    """Estimates term overlap and relevance between the question and the context."""
    q_words = set(w.lower().strip(".,!?") for w in question.split() if len(w) > 2)
    c_words = set(w.lower().strip(".,!?") for w in context.split() if len(w) > 2)

    if not q_words:
        return 0.0

    exact_matches = q_words & c_words
    exact_score = len(exact_matches) / len(q_words)

    partial_score = sum(
        1 for q in q_words
        for c in c_words
        if q[:4] == c[:4] and len(q) > 3
    ) / len(q_words) * 0.5

    return min(exact_score + partial_score, 1.0)

def fallback_response(question: str) -> str:
    """Message to return when RAG fails to retrieve any useful content."""
    return f"❌ No relevant information found in the knowledge base for: '{question}'"

def debug_rag_search(question: str) -> Dict:
    """Utility to debug Qdrant search behavior and thresholds."""
    if not collection_name:
        return {"error": "No Qdrant collection available."}

    try:
        preprocessed = preprocess_question(question)
        embedding_response = client.embeddings.create(
            input=[preprocessed],
            model=EMBEDDING_MODEL
        )
        query_vector = embedding_response.data[0].embedding

        thresholds_result = {}
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            hits = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=5,
                score_threshold=threshold
            )
            thresholds_result[f"score_threshold_{threshold}"] = len(hits)

        all_hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5
        )

        return {
            "collection": collection_name,
            "preprocessed_question": preprocessed,
            "results_by_threshold": thresholds_result,
            "top_hit_scores": [h.score for h in all_hits],
            "sample_payloads": [h.payload.get("page_content", "")[:100] for h in all_hits]
        }

    except Exception as e:
        return {"error": str(e)}

def classify_question(text):
    """Classify questions into SQL, RAG, or out-of-scope categories."""
    classification_prompt = """{classification_ques_prompt}
    
    Question: {question}"""
    
    try:
        response = client.chat.completions.create(
            model="ft:gpt-4.1-nano-2025-04-14:pharmaai:decision:Brj7DpRe:ckpt-step-76",
            messages=[
                {"role": "system", "content": classification_prompt.format(question=text)},
                {"role": "user", "content": text}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        category = response.choices[0].message.content.strip().lower()
        
        if "sql" in category:
            return "sql"
        elif "rag" in category:
            return "rag"
        else:
            return "out_of_scope"
            
    except Exception as e:
        logger.error(f"AI classification failed: {str(e)}, falling back to keyword matching")
        text = text.lower()
        sql_keywords = ["how many", "count", "list", "show", "patients", "filter", "average", "sum", "group by", "darzalex start", "who received"]
        rag_keywords = ["what is","types of", "explain", "used for", "describe", "define", "why", "purpose of", "meaning of", "cancer", "regimen overview", "are there any","what role does"]

        if any(k in text for k in sql_keywords):
            return "sql"
        elif any(k in text for k in rag_keywords):
            return "rag"
        return "out_of_scope"

def interpret_user_decision(text):
    """Interpret user confirmation or clarification"""
    normalized = text.lower().strip()
    yes_keywords = {
        "yes", "yeah", "yep", "correct", "right", "that's right", "sure", "go ahead",
        "umm yeah", "yo", "proceed", "generate", "do it", "exactly", "true", 
        "that works", "fine", "looks good", "yeh", "yea", "yeps"
    }
    return "yes" if any(kw in normalized for kw in yes_keywords) else "clarify"

# NEW END

# UPDATED
def validate_sql(sql):
    """Validate SQL is a safe SELECT query"""
    if sql.strip().startswith("```") and sql.strip().endswith("```"):
        sql = "\n".join(sql.strip().split("\n")[1:-1])

    clean_sql = " ".join(
        line for line in sql.splitlines()
        if not line.strip().startswith("--") and not line.strip().startswith("/*")
    ).strip().upper()

    return (
        ("SELECT" in clean_sql or clean_sql.startswith("WITH")) and
        "INSERT" not in clean_sql and
        "UPDATE" not in clean_sql and
        "DELETE" not in clean_sql and
        "DROP" not in clean_sql and
        "CREATE" not in clean_sql and
        "ALTER" not in clean_sql
    )


def log_feedback(original_question, clarified_question, interpretation, generated_sql, output_correct, feedback):
    """Log user feedback to Supabase"""
    supabase.table("feedback").insert({
        "original_question": original_question,
        "clarified_question": clarified_question,
        "ai_interpretation": interpretation,
        "generated_sql": generated_sql,
        "output_correct": output_correct,
        "feedback": feedback
    }).execute()

st.title("LLM PILOT")

if st.button("Clear Chat"):
    for key in ["messages", "phase", "original_query", "clarified_query", "last_sql", "last_interpretation", "feedback_given", "additional_feedback"]:
        st.session_state[key] = "" if "query" in key or "interpretation" in key else []
    st.rerun()

# UPDATED
for key in ["messages", "phase", "original_query", "clarified_query", "last_sql", "last_interpretation", "feedback_given", "additional_feedback"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "query" in key or "interpretation" in key else []

st.session_state.phase = st.session_state.phase or "waiting"

# UPDATED
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if isinstance(content, dict) and content.get("type") == "table":
            df = pd.DataFrame(content["data"])
            st.dataframe(df)
        else:
            st.markdown(content)

# Handle user input
user_input = st.chat_input("Ask or clarify your SQL question")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # NEW: Start
    if st.session_state.phase == "waiting":
        triage = classify_question(user_input)
    else:
        triage = st.session_state.get("last_triage", "sql")
    st.session_state.last_triage = triage
    

    st.write(f"Triage: {triage.upper()}")

    if triage == "sql":
        # New: end

        if st.session_state.phase == "waiting":
            # Initial question phase
            st.session_state.original_query = user_input
            st.session_state.clarified_query = ""
            st.session_state.last_sql = ""
            st.session_state.feedback_given = False
            st.session_state.additional_feedback = ""
            
            with st.chat_message("assistant"):
                with st.spinner("Let me try to understand your question..."):
                    interpretation = client.chat.completions.create(
                        model=model_interpretation,
                        messages=[
                            {"role": "system", "content": interpretation_prompt},
                            {"role": "user", "content": user_input}
                        ]
                    ).choices[0].message.content.strip()
                
                st.session_state.last_interpretation = interpretation
                msg = f"I understood your question as:\n\n**{interpretation}**\n\nShall I proceed with SQL generation, or would you like to refine your query?"
                st.markdown(msg)
            
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.phase = "confirm_intent"

        elif st.session_state.phase == "confirm_intent":

            # New Function
            decision = interpret_user_decision(user_input)

            if decision == "yes":
                query = st.session_state.clarified_query or st.session_state.original_query
                
                with st.chat_message("assistant"):
                    with st.spinner("Generating SQL..."):
                        sql_prompt = f"""The user asked: "{query}".
The AI understood it as: "{st.session_state.last_interpretation}"
The previous SQL attempt (if any): {st.session_state.last_sql or '[None]'}
Based on this, generate a correct SELECT query."""

                        try:
                            response = client.chat.completions.create(
                                model=model_id,
                                messages=[
                                    {"role": "system", "content": "Generate SQL queries for medical data."},
                                    {"role": "user", "content": sql_prompt}
                                ]
                            )
                            sql = response.choices[0].message.content.strip().rstrip(";")
                            st.session_state.last_sql = sql

                            # Always show the generated SQL
                            st.code(sql, language="sql")
                            st.session_state.messages.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})

                            if sql.strip().startswith("```") and sql.strip().endswith("```"):
                                sql = "\n".join(sql.strip().split("\n")[1:-1])
                            
                            sql = sql.strip()
                            if sql.endswith(";"):
                                sql = sql[:-1]
                            
                            if validate_sql(sql):
                                try:
                                    # Execute query
                                    result = supabase.rpc("run_raw_query", {"query": sql}).execute()

                                    if result.data:
                                        df = pd.DataFrame(result.data)
                                        if not df.empty:
                                            table_data = {
                                                "type": "table",
                                                "data": df.to_dict('records'),
                                                "columns": df.columns.tolist()
                                            }
                                            st.session_state.messages.append({"role": "assistant", "content": table_data})
                                            st.dataframe(df)
                                        else:
                                            st.info("Query executed successfully but returned no results.")
                                    else:
                                        st.info("Query executed but returned no data.")

                                    # Ask for confirmation
                                    followup = "Is this output correct?"
                                    st.markdown(followup)
                                    st.session_state.messages.append({"role": "assistant", "content": followup})
                                    st.session_state.phase = "confirm_result"

                                except Exception as e:
                                    error_msg = f"Query execution failed: {str(e)}"
                                    st.error(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                    log_feedback(
                                        st.session_state.original_query,
                                        st.session_state.clarified_query,
                                        st.session_state.last_interpretation,
                                        sql,
                                        False,
                                        str(e)
                                    )
                                    st.session_state.phase = "confirm_intent"
                            else:
                                error_msg = "Generated query isn't a valid SELECT statement. Please clarify your question."
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                log_feedback(
                                    st.session_state.original_query,
                                    st.session_state.clarified_query,
                                    st.session_state.last_interpretation,
                                    sql,
                                    False,
                                    "Invalid SELECT query"
                                )
                                st.session_state.phase = "confirm_intent"

                        except Exception as e:
                            error_msg = f"Failed to generate SQL: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.phase = "waiting"

            else:
                # Handle clarification
                st.session_state.clarified_query = user_input
                with st.chat_message("assistant"):
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
                                {"role": "system", "content": "Interpret the user's clarification about medical data."},
                                {"role": "user", "content": clarification_prompt}
                            ]
                        ).choices[0].message.content.strip()

                    st.session_state.last_interpretation = interpretation
                    msg = f"Got it. I understood your clarification as:\n\n**{interpretation}**\n\nShall I proceed with SQL generation, or would you like to refine your query?"
                    st.markdown(msg)
                
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.phase = "confirm_intent"

        elif st.session_state.phase == "confirm_result":
            # Result confirmation phase
            decision = interpret_user_decision(user_input)
            
            if decision == "yes":
                with st.chat_message("assistant"):
                    st.markdown("Great! Ask your next question anytime.")
                st.session_state.phase = "waiting"
            else:
                # Handle feedback or clarification
                st.session_state.clarified_query = user_input
                with st.chat_message("assistant"):
                    with st.spinner("Let me try to understand your clarification..."):
                        clarification_prompt = f"""The original question was:
{st.session_state.original_query}
The AI understood the original question as:
{st.session_state.last_interpretation}
The user provided feedback:
{user_input}
Please summarize what the user actually wants now."""
                        
                        interpretation = client.chat.completions.create(
                            model=model_interpretation,
                            messages=[
                                {"role": "system", "content": "Interpret the user's feedback about medical data."},
                                {"role": "user", "content": clarification_prompt}
                            ]
                        ).choices[0].message.content.strip()

                    st.session_state.last_interpretation = interpretation
                    msg = f"Got it. I understood your clarification as:\n\n**{interpretation}**\n\nShall I proceed with SQL generation, or would you like to refine your query?"
                    st.markdown(msg)
                
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.phase = "confirm_intent"

    # NEW START:
    elif triage == "rag":
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                rag_response = enhanced_rag_query(user_input)
                st.markdown(rag_response)
                st.session_state.messages.append({"role": "assistant", "content": rag_response})
        
        with st.chat_message("assistant"):
            followup_msg = "Feel free to ask any follow-up questions or ask about something else!"
            st.markdown(followup_msg)
            st.session_state.messages.append({"role": "assistant", "content": followup_msg})
        
        st.session_state.phase = "waiting"

    else:
        with st.chat_message("assistant"):
            out_msg = "❌ Sorry, this question is out of scope. Try asking about patient data or a known drug/regimen."
            st.markdown(out_msg)
            st.session_state.messages.append({"role": "assistant", "content": out_msg})
        st.session_state.phase = "waiting"
