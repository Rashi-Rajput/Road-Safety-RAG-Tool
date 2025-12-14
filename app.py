import os
import sys
import json
import re  
from typing_extensions import TypedDict
from typing import Annotated, List, Dict

from flask import Flask, render_template_string, request

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END, START


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: unable to find the api key")
    sys.exit(1)

try:
   
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0.0
    )
   

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
except Exception as e:
    print(f"Error {e}")
    sys.exit(1)

# Data feeding
def _setup_vector_store(csv_file: str) -> FAISS:
    try:
        loader = CSVLoader(
            file_path=csv_file,
            metadata_columns=['S. No.', 'code', 'clause'],
            encoding='utf-8'
        )
        data = loader.load()
    except Exception as e:
        print(f"Error in loading CSV data: {e}.")
        data = [Document(page_content="Error: Data Source unavailable.", metadata={'code': 'ERR', 'clause': '0'})]

  
    vectorstore = FAISS.from_documents(data, embeddings)
    return vectorstore

try:
  
    vector_store = _setup_vector_store("DATA SOURCE.csv")
    retriever = vector_store.as_retriever(k=4)
except Exception as e:
    print(f"Failed to create vector store: {e}")
    sys.exit(1)

# start the langgraph work by creating the state
class GraphState(TypedDict):
    question: str
    context: Annotated[List[Document], lambda x, y: x + y]
    answer: str
    relevance_grade: str

# definig nodes
# retrieve, grading, generate, poor
def retrieve(state: GraphState) -> Dict:
   
    question = state["question"]
    documents = retriever.invoke(question)
    
    context_data = []
    for doc in documents:
        source_ref = f"Source: {doc.metadata.get('code', 'N/A')}, Clause: {doc.metadata.get('clause', 'N/A')}"
        context_data.append(Document(
            page_content=doc.page_content,
            metadata={"source": source_ref}
        ))
    
    return {"context": context_data, "question": question, "relevance_grade": ""}

def grading(state: GraphState) -> Dict:
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    context_docs = state["context"]
    context_text = "\n---\n".join([f"Intervention Content:\n{doc.page_content}\n" for doc in context_docs])
#prompt for grading
    grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a strict document grader. Your task is to determine if the provided "
             "intervention suggestions (CONTEXT) are relevant to the user's road safety "
             "QUESTION. Output a single JSON object with the key 'relevance' and value 'relevant' "
             "if the context is useful, or 'irrelevant' otherwise. Be strict."
            ),
            ("human", 
             f"QUESTION: {question}\n\n"
             f"CONTEXT:\n{context_text}"
            )
        ]
    )

    response_str = llm.invoke(grader_prompt.format_messages(
        question=question, context=context_text
    )).content
    
    try:
        json_str = response_str.strip().replace('```json', '').replace('```', '')
        response_json = json.loads(json_str)
        grade = response_json.get("relevance", "irrelevant").lower()
    except Exception as e:
        print(f"Warning: Failed to parse JSON from grader: {e}. Defaulting to 'irrelevant'. Response was: {response_str}")
        grade = "irrelevant"
    
    print(f"Grading result: {grade}")
    return {"relevance_grade": grade}

def generate(state: GraphState) -> Dict:
    print("---NODE: GENERATE RESPONSE---")
    question = state["question"]
    context_docs = state["context"]
    context_text = "\n---\n".join([
        f"Intervention Suggestion:\n{doc.page_content}\nReference: {doc.metadata['source']}"
        for doc in context_docs
    ])
#prompt for final answer
    system_prompt = (
        "You are the ROAD SAFETY INTERVENTION GPT, an expert AI tool. "
        "Your task is to analyze the user's described road safety issue and the provided intervention suggestions. "
        "Based ONLY on the relevant context provided, you MUST select the most suitable intervention(s) "
        "and present your output in the following format:\n\n"
        "1. Recommended Intervention(s): State the recommended action(s).\n"
        "2. Explanation & Justification: Explain why this intervention is suitable for the described problem.\n"
        "3. Database Reference: Provide the exact 'Source' and 'Clause' from the reference text that supports your recommendation."
        "If multiple suggestions are combined, cite all relevant references. DO NOT make up interventions or references."
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", 
             f"CONTEXT:\n{context_text}\n\n"
             f"ROAD SAFETY ISSUE TO ADDRESS:\n{question}"
            )
        ]
    )

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": context_text, "question": question})
    return {"answer": response}


#message for irrelavent questions
def poor(state: GraphState) -> Dict:
    print("---NODE: POOR CONTEXT HANDLER---")
    message = (
        "**Road Safety Intervention GPT Status: Insufficient Data**\n\n"
        "I was unable to find specific, highly relevant road safety interventions in the database "
        "that directly address the issue you described. Please try rephrasing your road safety problem, "
        "or provide more context about the road type, specific hazard, or environment."
    )
    return {"answer": message}

# conditional 
def route_context(state: GraphState) -> str:
   
    grade = state["relevance_grade"]
    if grade == "relevant":
        
        return "generate"
    else:
        
        return "poor"

#  graph formation
def setup_rag_graph():
    """Sets up and compiles the LangGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grading", grading)
    workflow.add_node("generate", generate)
    workflow.add_node("poor", poor)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grading")
    
    workflow.add_conditional_edges(
        "grading",
        route_context,
        {
            "generate": "generate",
            "poor": "poor"
        }
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("poor", END)

    return workflow.compile()

# parsing the output
def parse_answer_into_three(answer_text: str) -> (str, str, str):
    
    if "**Road Safety Intervention GPT Status:**" in answer_text:
        return answer_text, "", ""

    intervention = re.search(r'1\.\s\*\*.+?\*\*(.*?)(?=\n2\.\s\*\*|\Z)', answer_text, re.DOTALL)
    explanation = re.search(r'2\.\s\*\*.+?\*\*(.*?)(?=\n3\.\s\*\*|\Z)', answer_text, re.DOTALL)
    reference = re.search(r'3\.\s\*\*.+?\*\*(.*)', answer_text, re.DOTALL)

    p1 = intervention.group(1).strip() if intervention else ""
    p2 = explanation.group(1).strip() if explanation else ""
    p3 = reference.group(1).strip() if reference else ""

    if not p1 and not p2 and not p3 and answer_text:
        return answer_text, "", ""
    
    return p1, p2, p3

#html part
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Intervention Gpt</title>
    <style>
       body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 2.5rem;
    background: linear-gradient(135deg, #eef2f7, #f8fafc);
    color: #2a2a2a;
}

.container {
    max-width: 850px;
    margin: 0 auto;
    background: #ffffff;
    padding: 2.5rem;
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.container:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 32px rgba(0,0,0,0.12);
}

h1, h2 {
    color: #0f172a;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}

form {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
}

textarea {
    font-family: inherit;
    font-size: 1rem;
    padding: 1rem;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    min-height: 120px;
    background: #f9fafb;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

textarea:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.15);
    outline: none;
}

button {
    font-size: 1rem;
    padding: 0.9rem;
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: background-color 0.25s ease, transform 0.15s ease;
    font-weight: 600;
    letter-spacing: 0.3px;
}

button:hover {
    background-color: #1d4ed8;
    transform: translateY(-2px);
}

button:active {
    transform: scale(0.97);
}

.results {
    margin-top: 2.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.75rem;
}

.output-box {
    background: #fdfdfd;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 6px 16px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}

.output-box:hover {
    transform: translateY(-2px);
}

h3 {
    margin-top: 0;
    color: #1d4ed8;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.5rem;
    font-size: 1.15rem;
}

pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: "SF Mono", "Consolas", monospace;
    font-size: 0.95rem;
    color: #374151;
    line-height: 1.6;
}

    </style>
</head>
<body>
    <div class="container">
        <h1>Road Safety Intervention GPT</h1>
        <p>Describe a road safety issue to get analysis and recommendations from the database.</p>
        
        <form action="/process" method="POST">
            <label for="question"><strong>Describe the issue:</strong></label>
            <textarea name="question" id="question" rows="5" placeholder="Enter Your Intervention">{{ question }}</textarea>
            <button type="submit">Analyze Issue</button>
        </form>

       
        {% if output1 or output2 or output3 %}
        <div class="results">
            <h2>Analysis Results</h2>

            <div class="output-box">
                <h3>Recommended Intervention(s)</h3>
                <!-- If it's an error message, it will appear here -->
                <pre>{{ output1 }}</pre>
            </div>

            <!-- Only show boxes 2 and 3 if they have content -->
            {% if output2 %}
            <div class="output-box">
                <h3>Explanation & Justification</h3>
                <pre>{{ output2 }}</pre>
            </div>
            {% endif %}
            
            {% if output3 %}
            <div class="output-box">
                <h3>Database Reference</h3>
                <pre>{{ output3 }}</pre>
            </div>
            {% endif %}

        </div>
        {% endif %}
    </div>
</body>
</html>
"""


app = Flask(__name__)
compiled_rag_app = setup_rag_graph()




@app.route('/')
def index():
    
    return render_template_string(
        HTML_TEMPLATE, 
        question="", 
        output1="", 
        output2="", 
        output3=""
    )

@app.route('/process', methods=['POST'])
def process():
    
    user_question = request.form['question']
    
    if not user_question:
        return render_template_string(
            HTML_TEMPLATE, 
            question="", 
            output1="Please enter a question.", 
            output2="", 
            output3=""
        )

    print(f"\n--- PROCESSING NEW REQUEST: {user_question} ---")
    
    input_data = {"question": user_question}
    
    try:
        final_state = compiled_rag_app.invoke(input_data)
        raw_answer = final_state['answer']
        
       
        print(raw_answer)
       
        out1, out2, out3 = parse_answer_into_three(raw_answer)

       
        return render_template_string(
            HTML_TEMPLATE, 
            question=user_question, 
            output1=out1, 
            output2=out2, 
            output3=out3
        )
        
    except Exception as e:
        print(f"Error during graph execution: {e}")
        return render_template_string(
            HTML_TEMPLATE, 
            question=user_question, 
            output1=f"An error occurred: {e}", 
            output2="", 
            output3=""
        )


if __name__ == '__main__':
   
    app.run(debug=True, host='0.0.0.0', port=5000)
