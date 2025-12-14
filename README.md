# Road Safety Intervention GPT 🚦

Road Safety Intervention GPT is an intelligent web-based application that analyzes road safety issues and recommends suitable interventions using a Retrieval-Augmented Generation (RAG) architecture. The system ensures accurate, explainable, and reference-backed recommendations sourced from a structured domain-specific database.

## 🔍 Features
- Uses Google Gemini LLM for reasoning and response generation
- Retrieval-Augmented Generation (RAG) with FAISS vector store
- Strict relevance grading to avoid hallucinated answers
- Graph-based workflow using LangGraph
- Clear fallback handling for insufficient data
- Clean and responsive Flask-based web interface

## 🧠 Architecture
1. **User Input**: Road safety issue description
2. **Document Retrieval**: Similarity search over CSV-based intervention database
3. **Relevance Grading**: Filters irrelevant context
4. **Response Generation**: Produces structured output:
   - Recommended Intervention(s)
   - Explanation & Justification
   - Database Reference (Source & Clause)

## 🗂️ Data Source
- Intervention data stored in a CSV file (`DATA SOURCE.csv`)
- Embedded using Google Generative AI Embeddings
- Indexed with FAISS for fast retrieval

## 🛠️ Tech Stack
- Python
- Flask
- LangChain
- LangGraph
- Google Gemini API
- FAISS
- HTML & CSS

## Instructions 

# 1. Clone the repository
git clone https://github.com/Rashi-Rajput/road-safety-intervention-gpt.git
cd road-safety-intervention-gpt


# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
 Create a .env file and add:
 GOOGLE_API_KEY=your_google_api_key_here

# 5. Run the application
python app.py

# 6. Open in browser
# http://localhost:5000
