
# Tool Assisted Agent

A chat agent built with FastAPI and LangGraph. It supports academic assistance, emotional understanding, safety protocols, and conversation memory across threads.

## Features

- 🤖 Smart routing to specialized handlers
- 📚 Academic utilities such as grade and average calculations
- ❤️ Positive and negative emotional response handling
- 🛡️ Automatic safety alerts for self-harm concerns
- 💾 Persistent memory across conversations
- 🧵 Multi-threaded chat sessions

## Requirements

- Python 3.8+
- Groq API key

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd langgraph-chat-api
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Create environment file
cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the service

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| URL       | Method | Description                  |
| --------- | ------ | ---------------------------- |
| `/docs`   | GET    | OpenAPI documentation        |
| `/health` | GET    | Health status of the service |

Once running:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

