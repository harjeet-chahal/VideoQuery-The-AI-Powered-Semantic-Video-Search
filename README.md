# VideoQuery - Multimodal RAG Application

A Multimodal RAG (Retrieval Augmented Generation) application that allows users to upload videos, search visual and audio content, and answer questions about the video.

## Features

- 🎬 **Video Processing**: Extract audio, frames, and generate embeddings
- 🎤 **Audio Transcription**: Transcribe audio using OpenAI Whisper
- 🖼️ **Visual Search**: Search video frames using CLIP embeddings
- 💬 **Natural Language Queries**: Ask questions about video content
- 🤖 **AI-Powered Answers**: Generate answers using Llama 3 (via Groq or Ollama)
- ⏱️ **Timestamp Citations**: View specific frames at mentioned timestamps

## Tech Stack

- **Backend**: Python, FastAPI
- **AI Processing**: OpenAI Whisper (Audio), OpenAI CLIP (Vision), Llama 3 (Reasoning)
- **Database**: ChromaDB (Local persistence)
- **Frontend**: Streamlit

## Setup

1. **Install dependencies**:
   ```bash
   ./setup.sh
   # Or manually:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

## Running the Application

### Start the Backend API

Option 1: Run from project root (recommended)
```bash
# From the VideoQuery project root directory
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Option 2: Run from backend directory
```bash
cd backend
python api.py
```

The API will run on `http://localhost:8000`

### Start the Frontend

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload a video**: Use the sidebar to upload an MP4 video file
2. **Process the video**: Click "Process Video" to extract frames, transcribe audio, and generate embeddings
3. **Ask questions**: Type questions about the video content in the chat interface
4. **View results**: See AI-generated answers with relevant frames displayed automatically

## Project Structure

```
VideoQuery/
├── backend/
│   ├── api.py          # FastAPI backend endpoints
│   ├── ingest.py       # Video processing pipeline
│   ├── database.py     # ChromaDB management
│   └── rag.py          # RAG query system
├── frontend/
│   └── app.py          # Streamlit UI
├── data/
│   ├── chromadb/       # ChromaDB persistent storage
│   ├── uploads/        # Uploaded video files
│   └── processed/      # Processed video data
├── requirements.txt
├── setup.sh
└── README.md
```

## Next Steps

- Implement actual Llama 3 API integration (Groq or Ollama) in `backend/rag.py`
- Add video playback controls
- Support for multiple videos
- Export query results

