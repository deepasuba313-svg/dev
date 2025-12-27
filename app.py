import pandas as pd
from flask import Flask, request,render_template
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
app=Flask(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
df = pd.read_csv("/workspaces/dev/qa_data (1).csv")
context_text = ""
for _, row in df.iterrows():
    context_text += f"Q: {row['question']}\nA: {row['answer']}\n\n"

def ask_gemini(query):
    prompt = f"""
You are a Q&A assistant.
Use the following context to answer the question.
If the answer is not in the context, respond with:
"Irrelevant question and answer".
Context:
{context_text}
Question: {query}
Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        user_query = request.form["query"]
        answer = ask_gemini(user_query)
    return render_template("index.html", answer=answer) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
#while True:
 #   user_query = input("Enter your question (or type 'exit' to quit): ")
  #  if user_query.lower() == 'exit':
   #     break

    #answer = ask_gemini(user_query)
 #   print(f"Answer: {answer}\n")
