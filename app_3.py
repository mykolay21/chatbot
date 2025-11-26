import streamlit as st
import os
import psycopg2
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

MODEL = "gpt-4o"

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-04-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    timeout=60
)


# Database connection function
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        database=os.environ.get("PGDATABASE", "postgres"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        port=os.environ.get("PGPORT", "5432")
    )


# Function definitions for AI
functions = [
    {
        "name": "query_database",
        "description": "Execute a SQL query on the PostgreSQL database and return the results",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_table_schema",
        "description": "Get the schema of database tables to understand their structure",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to get schema for (optional)"
                }
            },
            "required": []
        }
    }
]

# Available functions for the AI to call
available_functions = {
    "query_database": lambda query: execute_sql_query(query),
    "get_table_schema": lambda table_name=None: get_table_schema(table_name)
}


def execute_sql_query(query):
    """Execute SQL query and return results"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query)

        if query.strip().lower().startswith(('select', 'show', 'describe')):
            result = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return {"columns": columns, "data": result, "row_count": len(result)}
        else:
            conn.commit()
            return {"message": f"Query executed successfully. Rows affected: {cur.rowcount}"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


def get_table_schema(table_name=None):
    """Get schema information for tables"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if table_name:
            # Get schema for specific table
            cur.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, (table_name,))
            result = cur.fetchall()
            return {"table": table_name, "schema": result}
        else:
            # Get all tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cur.fetchall()
            return {"tables": [table[0] for table in tables]}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


def process_user_input(user_input, conversation_history):
    """Process user input using OpenAI with function calling"""

    messages = conversation_history + [{"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    response_message = response.choices[0].message

    # Check if function call is needed
    if response_message.function_call:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)

        st.sidebar.write(f"🔄 Calling function: {function_name}")
        st.sidebar.write(f"Arguments: {function_args}")

        # Execute the function
        function_to_call = available_functions[function_name]
        if function_name == "get_table_schema":
            function_response = function_to_call(function_args.get('table_name'))
        else:
            function_response = function_to_call(**function_args)

        # Add function response to conversation
        messages.append({
            "role": "function",
            "name": function_name,
            "content": json.dumps(function_response, default=str)
        })

        # Get final response from AI
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )

        return second_response.choices[0].message.content, messages + [
            {"role": "function", "name": function_name, "content": json.dumps(function_response, default=str)},
            {"role": "assistant", "content": second_response.choices[0].message.content}
        ]

    return response_message.content, messages + [{"role": "assistant", "content": response_message.content}]


# Streamlit UI
st.set_page_config(page_title="Database Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Database Chatbot")
st.markdown("Chat with your PostgreSQL database using natural language!")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system",
         "content": "You are a helpful assistant that can interact with a PostgreSQL database. Always be concise and helpful."}
    ]

# Sidebar for database info and examples
with st.sidebar:
    st.header("Database Info")

    try:
        schema_info = get_table_schema()
        if "tables" in schema_info:
            st.write(f"**Tables in database:** {', '.join(schema_info['tables'])}")
        elif "error" in schema_info:
            st.error(f"Database connection error: {schema_info['error']}")
    except Exception as e:
        st.error(f"Could not connect to database: {e}")

    st.header("Example Queries")
    st.markdown("""
    - "What tables are in the database?"
    - "Show me the schema of the users table"
    - "How many records are in the products table?"
    - "Select all data from the customers table"
    """)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about your database..."):
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process user input
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, updated_messages = process_user_input(user_input, st.session_state.messages)
            st.markdown(response)

    # Update conversation history
    st.session_state.messages = updated_messages

# Add some usage tips
with st.expander("💡 Usage Tips"):
    st.markdown("""
    - The chatbot can execute SQL queries and read database schemas
    - Be specific about which tables or data you're interested in
    - For sensitive operations, ensure proper permissions are set
    - Complex queries might take longer to process
    - The bot will automatically understand your database structure
    """)