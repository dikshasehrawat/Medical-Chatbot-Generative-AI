🏥 Medical Chatbot – Generative AI

An AI-powered medical chatbot that answers health-related queries using Generative AI and Retrieval-Augmented Generation (RAG).
The chatbot retrieves relevant information from medical documents and generates contextual responses to assist users with healthcare-related questions.

📌 Project Description

Built an AI-powered medical chatbot that provides healthcare information using natural language conversations.

Implemented Retrieval-Augmented Generation (RAG) to improve response accuracy by retrieving knowledge from medical documents.

Used vector embeddings and semantic search to find relevant information before generating answers.

Integrated LLM-based text generation to produce context-aware medical responses.

Designed a web-based interface allowing users to interact with the chatbot easily.

🚀 Features

💬 Conversational AI chatbot for medical queries

📚 Retrieval-Augmented Generation for accurate answers

🔎 Semantic search using vector embeddings

⚡ Fast response generation using LLMs

🌐 Simple web interface for user interaction

🧠 Knowledge retrieval from medical documents

🛠 Tech Stack

Python

LangChain

OpenAI GPT

Pinecone (Vector Database)

Flask

HTML / CSS

These tools help build conversational AI systems that retrieve knowledge and generate responses using large language models.

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/dikshasehrawat/Medical-Chatbot-Generative-AI.git
cd Medical-Chatbot-Generative-AI
2️⃣ Create a virtual environment
conda create -n medibot python=3.10 -y
conda activate medibot
3️⃣ Install dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory and add:

PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
▶️ Run the Application
python store_index.py
python app.py

Then open:

http://localhost:5000
📂 Project Structure
Medical-Chatbot-Generative-AI
│
├── Data/                # Medical dataset
├── src/                 # Core chatbot logic
├── templates/           # Frontend HTML
├── static/              # CSS and assets
├── app.py               # Main application
├── store_index.py       # Embedding generation
├── requirements.txt     # Dependencies
└── README.md
🎯 Future Improvements

Add voice-based medical assistant

Improve medical dataset coverage

Add multi-language support

Deploy using Docker + Cloud services

Integrate authentication and chat history

⚠️ Disclaimer

This chatbot is intended for educational and informational purposes only and should not replace professional medical advice.

Diksha Sehrawat

✅ This README will make your repo look professional and recruiter-friendly.
