"""
FastAPI Backend for VideoQuery
Handles video upload, processing, and query endpoints.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from dotenv import load_dotenv
import os
import uuid
import shutil

from backend.ingest import VideoProcessor
from backend.database import VideoDatabase, initialize_database
from backend.rag import VideoRAG

app = FastAPI(title="VideoQuery API")
load_dotenv(dotenv_path=project_root / ".env")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize database and models (lazy loading)
db = None
processor = None
rag = None

def get_database():
    """Lazy initialization of database."""
    global db
    if db is None:
        db = initialize_database()
    return db

def get_processor():
    """Lazy initialization of video processor."""
    global processor
    if processor is None:
        processor = VideoProcessor(whisper_model_size="small")
    return processor

def get_rag():
    """Lazy initialization of RAG system."""
    global rag
    if rag is None:
        rag = VideoRAG(database=get_database())
    return rag


class QueryRequest(BaseModel):
    query: str
    video_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    context: str
    frame_results: Optional[Dict] = None
    transcript_results: Optional[Dict] = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "VideoQuery API is running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload and process a video file.
    
    Returns:
        Dictionary with video_id and processing status
    """
    # Validate file type
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Process video
    try:
        processor = get_processor()
        output_dir = os.path.join("./data/processed", video_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process video
        result = processor.process_video(file_path, output_dir=output_dir)
        
        # Store in database
        db = get_database()
        counts = db.store_video_data(video_id, result)
        
        return {
            "video_id": video_id,
            "status": "processed",
            "frames_stored": counts["frames_stored"],
            "transcripts_stored": counts["transcripts_stored"],
            "frames": result["frames"],  # Include frame paths and timestamps
            "message": "Video processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    """
    Query video content using RAG.
    
    Returns:
        Answer and context from RAG system
    """
    try:
        rag = get_rag()
        result = rag.query_video(
            user_query=request.query,
            video_id_filter=request.video_id
        )
        
        return QueryResponse(
            answer=result["answer"],
            context=result["context"],
            frame_results=result.get("frame_results"),
            transcript_results=result.get("transcript_results")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying video: {str(e)}")


@app.get("/video/{video_id}/frames")
def get_video_frames(video_id: str):
    """
    Get all frames for a specific video.
    
    Returns:
        Dictionary with frame information
    """
    try:
        db = get_database()
        frames = db.get_video_frames(video_id)
        return frames
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving frames: {str(e)}")


@app.get("/frame/{frame_path:path}")
def get_frame_image(frame_path: str):
    """
    Get frame image file.
    Note: In production, use proper file serving or cloud storage.
    """
    if os.path.exists(frame_path):
        from fastapi.responses import FileResponse
        return FileResponse(frame_path)
    else:
        raise HTTPException(status_code=404, detail="Frame not found")


@app.get("/video/{video_id}/transcript")
def get_video_transcript(video_id: str):
    """
    Get all transcript segments for a specific video.
    
    Returns:
        Dictionary with transcript segments
    """
    try:
        db = get_database()
        transcripts = db.get_video_transcripts(video_id)
        return transcripts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transcript: {str(e)}")


@app.get("/summary/{video_id}")
def get_video_summary(video_id: str):
    """
    Generate a summary of the video by retrieving all transcripts and summarizing with Groq.
    
    Returns:
        Dictionary with summary text
    """
    try:
        # Get all transcript segments for the video
        db = get_database()
        transcripts = db.get_video_transcripts(video_id)
        
        if not transcripts or not transcripts.get("documents"):
            raise HTTPException(
                status_code=404, 
                detail=f"No transcript found for video_id: {video_id}"
            )
        
        # Stitch together all transcript segments
        documents = transcripts.get("documents", [])
        metadatas = transcripts.get("metadatas", [])
        
        # Sort by timestamp_start to maintain chronological order
        transcript_parts = []
        for doc, metadata in zip(documents, metadatas):
            start_time = metadata.get("timestamp_start", 0.0)
            end_time = metadata.get("timestamp_end", 0.0)
            transcript_parts.append((start_time, end_time, doc))
        
        # Sort by start time
        transcript_parts.sort(key=lambda x: x[0])
        
        # Combine all transcript segments
        full_transcript = "\n".join([doc for _, _, doc in transcript_parts])
        
        # Generate summary using Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable is not set"
            )
        
        from groq import Groq
        groq_client = Groq(api_key=groq_api_key)
        
        system_prompt = (
            "You are a helpful video assistant. Summarize the key takeaways from the video transcript."
        )
        
        user_prompt = f"""Video Transcript:
{full_transcript}

Please provide a concise summary of the key takeaways from this video."""
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        summary = response.choices[0].message.content.strip()
        
        return {
            "video_id": video_id,
            "summary": summary,
            "transcript_length": len(full_transcript),
            "num_segments": len(transcript_parts)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

