import streamlit as st
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

MODEL = "gpt-4o"


# -------- PostgreSQL CONNECTION ----------
def execute_sql_query(sql: str):
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        dbname=os.getenv("PG_DB", "analytics_chatbot"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "password"),
        port=os.getenv("PG_PORT", 5432),
        cursor_factory=RealDictCursor
    )
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows


client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-04-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    timeout=60
)


# -----------------------------
# TOOL DEFINITION
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Run a SQL query on PostgreSQL and return results as a list of rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        }
    }
]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("PostgreSQL Chatbot with Function Calling")

question = st.text_input("Ask a question about your database:")

if st.button("Run") and question:

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that converts natural language questions into SQL. "
                "Only call the SQL execution function when appropriate. "
                "Never write SQL that modifies data. SELECT only."
            )
        },
        {"role": "user", "content": question},
    ]

    # -------------------------
    # FIRST CALL — model decides whether to call the SQL function
    # -------------------------
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # If the model decided to call a tool:
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        if tool_call.function.name == "execute_sql_query":
            # arguments is a JSON string → convert to dict
            args = json.loads(tool_call.function.arguments)
            sql_query = args["sql"]

            # Execute SQL on backend
            try:
                rows = execute_sql_query(sql_query)
            except Exception as e:
                st.error(str(e))
                rows = {"error": str(e)}

            # Send tool result back to LLM
            messages.append(assistant_message)  # include tool call info
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(rows),
            })

            # NOW LLM PROVIDES FINAL ANSWER WITH RESULTS
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )

            st.write(final_response.choices[0].message.content)

    else:
        # LLM answered normally without tool call
        st.write(assistant_message.content)
