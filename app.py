import streamlit as st
import os
import re
import psycopg2
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

MODEL = "gpt-4o"

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-04-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    timeout=60  # ← FIX
)


# -------- PostgreSQL CONNECTION ----------
def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        database=os.getenv("PG_DB", "analytics_chatbot"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "password"),
        port=os.getenv("PG_PORT", 5432)
    )


# -------- LLM PROMPTS ----------
SQL_PROMPT = """
You are a PostgreSQL SQL generator.
Convert the user question into a safe SQL query.
Return only SQL. No explanation.
Available tables:
customers(customer_id, name, country)
products(product_id, name, category, price)
orders(order_id, customer_id, order_date, total_amount)
order_items(order_id, product_id, quantity)
Important rule that prohibits any DML operations such as INSERT, UPDATE, or DELETE
"""

SUMMARY_PROMPT = """
You are a data assistant.
Summarize the SQL result in 1–3 sentences.
User question, the SQL used, and the query results will be provided.
"""

# -------- STREAMLIT UI ----------
st.title("PostgreSQL LLM Chatbot")
st.caption("Ask questions about your data. The LLM writes SQL → runs it → summarizes results.")

# Initialize chat history. Every time the app is reloaded, the chat history will be saved in the session state:
# st.session_state is a dictionary-like object that allows you to store information across multiple runs of the app.
# At the first run there is no "messages" key in the session state, so we initialize it with a system prompt:
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Display chat messages from history on app rerun. Every time the app is reloaded, the chat history will be displayed.
# st.chat_message is a container that can be used to display chat messages in a chat-like interface.
# The role parameter can be "user" or "assistant" to differentiate between user and assistant messages.
for message in st.session_state.messages:
    if message["role"] == "system":
        continue  # Skip displaying system messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input. The first time the app is run, this will be empty.
# When the user enters a message, the app will be reloaded and the message will be processed.
# st.chat_input is a text input box that can be used to accept user input in a chat-like interface.
# When the user submits a message, the prompt variable will contain the message text.
prompt = st.chat_input("How can I help you?")
if prompt:
    # Add user message to chat history. If don't add it, it will be lost when the app is reloaded.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # ====== 1. Generate SQL ======
    sql_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SQL_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    # sql_query = sql_response.choices[0].message.content.strip()
    #  sql_query = re.sub(r'```(?:sql)?\s*|\s*```', '', sql_response.choices[0].message.content.strip(),
    #                    flags=re.IGNORECASE)

    raw_sql = sql_response.choices[0].message.content.strip()


    # ── Robust SQL extraction (handles multiple common patterns) ──
    def extract_sql(text: str) -> str:
        """
        Extract SQL from LLM output, removing Markdown fences, explanations, etc.
        Returns clean SQL or raises a clear error.
        """
        # Remove Markdown code blocks (```sql ... ``` or ``` ... ```)
        sql = re.sub(r'^```(?:sql)?\s*', '', text, flags=re.IGNORECASE)
        sql = re.sub(r'```$', '', sql)

        # Remove any leading/trailing explanations (common in GPT-4o)
        # Keep only the first valid SQL statement if multiple are returned
        sql = sql.strip()

        # Basic safety: ensure it starts with a SQL keyword
        if not re.match(r'^\s*(SELECT|WITH|EXPLAIN|SHOW|PRAGMA)', sql, re.IGNORECASE):
            raise ValueError("Generated text does not appear to be a valid SQL query")

        # Optional but recommended: limit length to prevent injection/runaway queries
        if len(sql) > 2000:
            raise ValueError("Generated SQL is too long")

        return sql


    try:
        sql_query = extract_sql(raw_sql)
    except Exception as extract_err:
        sql_query = f"Failed to extract clean SQL: {extract_err}"
        rows = None
    else:
        st.code(sql_query, language="sql")  # now safe to display

        # ====== 2. Execute SQL safely ======
        conn = get_pg_connection()
        cur = conn.cursor()
        try:
            cur.execute(sql_query)
            if sql_query.strip().upper().startswith("SELECT"):
                rows = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                # Convert to list of dicts for nicer summary
                rows = [dict(zip(column_names, row)) for row in rows]
            else:
                conn.commit()
                rows = f"Query executed successfully. Rows affected: {cur.rowcount}"
        except Exception as e:
            rows = f"SQL Execution Error: {str(e)}"
        finally:
            cur.close()
            conn.close()

    # Show the generated SQL (debugging)
    st.code(sql_query, language="sql")

    # ====== 2. Run SQL against PostgreSQL ======
    conn = get_pg_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
    except Exception as e:
        rows = f"SQL Error: {str(e)}"
    conn.commit()
    cur.close()
    conn.close()

    # ====== 3. Summarize results ======
    summary_payload = f"""
    User question: {prompt}
    SQL executed: {sql_query}
    Result: {rows}
    """
    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": summary_payload},
        ],
    )
    summary = summary_response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": summary})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(summary)
