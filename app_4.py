import streamlit as st
from openai import AzureOpenAI
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
import json

load_dotenv()

# ============================
# Configuration
# ============================
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    timeout=60
)

MODEL = "gpt-4o"


# PostgreSQL connection (read-only recommended)
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        database=os.getenv("PG_DB", "analytics_chatbot"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "password"),
        port=os.getenv("PG_PORT", 5432),
        cursor_factory=RealDictCursor,
    )


# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="SQL Chatbot", page_icon="🗄️")
st.title("🗄️ Natural Language → SQL Chatbot")
st.caption("Ask questions about your PostgreSQL database in plain English")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system",
         "content": "You are a helpful SQL assistant. Use the provided function to query the database."}
    ]

# Display chat history
for msg in st.session_state.messages[1:]:  # skip system message
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================
# Function definition for Azure OpenAI
# ============================
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute a read-only SQL query on the PostgreSQL database and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute. Must be SELECT only. Limit results with LIMIT 100."
                    }
                },
                "required": ["sql"],
                "additionalProperties": False
            }
        }
    }
]

# ============================
# Chat input
# ============================
if prompt := st.chat_input("What would you like to know from the database?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Call Azure OpenAI with function calling
        response = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            stream=False  # We'll stream manually below if needed
        )

        message = response.choices[0].message

        # If LLM wants to call the function
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "execute_sql_query":
                sql = arguments["sql"].strip()

                # Safety: block dangerous commands
                sql_lower = sql.lower()
                if any(cmd in sql_lower for cmd in ["insert", "update", "delete", "drop", "create", "alter", "grant"]):
                    result = "Error: Only SELECT queries are allowed."
                else:
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(sql)
                        rows = cur.fetchall()
                        columns = [desc[0] for desc in cur.description] if cur.description else []

                        if not rows:
                            result = "Query executed successfully. No rows returned."
                        else:
                            # Format nicely
                            import pandas as pd

                            df = pd.DataFrame(rows, columns=columns)
                            result = f"Found {len(df)} rows:\n\n{df.head(100).to_markdown(index=False)}"
                            if len(df) > 100:
                                result += f"\n\n... and {len(df) - 100} more rows (showing first 100)"
                    except Exception as e:
                        import traceback
                        error_text = traceback.format_exc()
                        st.error(error_text)  # shows real error in UI
                        result = "SQL execution failed."
                    finally:
                        if 'cur' in locals():
                            cur.close()
                        if 'conn' in locals():
                            conn.close()

                # Append tool result and get final answer
                # Convert message to dict before appending
                st.session_state.messages.append({
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": getattr(message, "tool_calls", None)  # if you need tool_calls
                })
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })

                # Second call: let LLM summarize the result
                second_response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=0.3
                )
                final_message = second_response.choices[0].message.content
                full_response = final_message

        else:
            full_response = message.content or "Sorry, I couldn't process that."

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
