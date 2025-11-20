from fastapi import FastAPI
from openai import OpenAI
import psycopg2
import os

client = OpenAI()

app = FastAPI()

# PostgreSQL connection
def get_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        database=os.getenv("PG_DB", "ecommerce"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "password"),
        port=os.getenv("PG_PORT", 5432)
    )

SQL_SYSTEM_PROMPT = """
You convert user questions into SQL queries for an e-commerce PostgreSQL database.
Return only SQL. No explanations.
Tables:
customers(customer_id, name, country)
products(product_id, name, category, price)
orders(order_id, customer_id, order_date, total_amount)
order_items(order_id, product_id, quantity)
"""

SUMMARY_SYSTEM_PROMPT = """
You summarize database query results for an analytics user.
Write a short, clean answer.
"""

def llm_generate_sql(question: str):
    response = client.responses.create(
        model="gpt-4.1",
        system=SQL_SYSTEM_PROMPT,
        input=[{"role": "user", "content": question}]
    )
    return response.output_text.strip()

def llm_summarize(question, sql, rows):
    payload = f"""
User question: {question}
SQL executed: {sql}
Result: {rows}
"""
    response = client.responses.create(
        model="gpt-4.1",
        system=SUMMARY_SYSTEM_PROMPT,
        input=[{"role": "user", "content": payload}]
    )
    return response.output_text.strip()

@app.get("/ask")
def ask(question: str):
    sql = llm_generate_sql(question)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()

    summary = llm_summarize(question, sql, rows)

    return {
        "answer": summary,
        "sql": sql,
        "rows": rows
    }
