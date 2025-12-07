"""
Streamlit Frontend for VideoQuery
Multimodal RAG application for video content querying.
"""

import streamlit as st
import requests
import re
import os
from pathlib import Path
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="VideoQuery",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "frames_data" not in st.session_state:
    st.session_state.frames_data = {}
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = {}


def extract_timestamps(text: str) -> list:
    """
    Extract timestamps from text in format MM:SS or HH:MM:SS.
    
    Returns:
        List of timestamp strings found in text
    """
    # Pattern for MM:SS or HH:MM:SS
    pattern = r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b'
    timestamps = re.findall(pattern, text)
    return timestamps


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string (MM:SS or HH:MM:SS) to seconds.
    
    Args:
        timestamp: Timestamp string like "02:14" or "1:02:14"
    
    Returns:
        Timestamp in seconds
    """
    parts = timestamp.split(':')
    if len(parts) == 2:
        # MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # HH:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    return 0.0


def find_closest_frame(timestamp_seconds: float, frames_data: dict) -> str:
    """
    Find the frame path closest to the given timestamp.
    
    Args:
        timestamp_seconds: Timestamp in seconds
        frames_data: Dictionary with frame information
    
    Returns:
        Path to the closest frame image
    """
    if not frames_data or "metadatas" not in frames_data:
        return None
    
    metadatas = frames_data.get("metadatas", [])
    ids = frames_data.get("ids", [])
    
    if not metadatas or not ids:
        return None
    
    # Find frame with closest timestamp
    closest_frame = None
    min_diff = float('inf')
    
    for frame_id, metadata in zip(ids, metadatas):
        frame_timestamp = metadata.get("timestamp", 0.0)
        diff = abs(frame_timestamp - timestamp_seconds)
        
        if diff < min_diff:
            min_diff = diff
            closest_frame = metadata.get("frame_path")
    
    return closest_frame


def process_video(file):
    """Process uploaded video file."""
    try:
        # Upload file to backend
        files = {"file": (file.name, file, "video/mp4")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.video_id = result["video_id"]
            st.session_state.frames_data = {
                "frames": result.get("frames", []),
                "video_id": result["video_id"]
            }
            return result
        else:
            st.error(f"Error processing video: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Please make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def query_video(query: str, video_id: str = None):
    """Query video content."""
    try:
        payload = {"query": query, "video_id": video_id}
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error querying video: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Please make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# Main UI
st.title("🎬 VideoQuery")
st.markdown("### Multimodal RAG Application for Video Content")

# Sidebar for video upload
with st.sidebar:
    st.header("📤 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4"],
        help="Upload an MP4 video file to process"
    )
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("🔄 Process Video", type="primary", use_container_width=True):
            with st.spinner("Processing video... This may take a few minutes."):
                result = process_video(uploaded_file)
                
                if result:
                    st.success("✅ Video processed successfully!")
                    st.info(f"**Video ID:** {result['video_id']}")
                    st.info(f"**Frames stored:** {result['frames_stored']}")
                    st.info(f"**Transcripts stored:** {result['transcripts_stored']}")
                else:
                    st.error("Failed to process video.")

# Main content area
if st.session_state.video_id:
    st.success(f"✅ Video loaded (ID: {st.session_state.video_id})")
    
    # Chat interface
    st.header("💬 Ask Questions About the Video")
    
    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Show chat history
    for i, (role, message, answer, context) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.write(message)
            if answer:
                st.markdown("**Answer:**")
                st.write(answer)
                
                # Extract and display frames for timestamps
                timestamps = extract_timestamps(answer)
                if timestamps and st.session_state.frames_data:
                    st.markdown("**Relevant Frames:**")
                    cols = st.columns(min(len(timestamps), 3))
                    
                    for idx, timestamp in enumerate(timestamps):
                        timestamp_seconds = timestamp_to_seconds(timestamp)
                        frame_path = find_closest_frame(
                            timestamp_seconds,
                            st.session_state.frames_data
                        )
                        
                        if frame_path and os.path.exists(frame_path):
                            with cols[idx % len(cols)]:
                                try:
                                    img = Image.open(frame_path)
                                    st.image(img, caption=f"At {timestamp}", use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not load frame: {e}")
    
    # Chat input
    user_query = st.chat_input("Ask a question about the video...")
    
    if user_query:
        # Add user message to chat
        st.session_state.chat_history.append(("user", user_query, None, None))
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Query video
        with st.spinner("Searching video content and generating answer..."):
            result = query_video(user_query, st.session_state.video_id)
        
        if result:
            answer = result.get("answer", "")
            context = result.get("context", "")
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown("**Answer:**")
                st.write(answer)
                
                # Extract and display frames for timestamps in answer
                timestamps = extract_timestamps(answer)
                if timestamps and st.session_state.frames_data.get("frames"):
                    st.markdown("**📸 Relevant Frames:**")
                    
                    # Get frames from session state
                    frames_list = st.session_state.frames_data.get("frames", [])
                    frame_dict = {timestamp: path for path, timestamp in frames_list}
                    
                    # Display frames for found timestamps
                    cols = st.columns(min(len(timestamps), 3))
                    
                    for idx, timestamp_str in enumerate(timestamps):
                        timestamp_seconds = timestamp_to_seconds(timestamp_str)
                        
                        # Find closest frame
                        closest_frame = None
                        min_diff = float('inf')
                        
                        for frame_path, frame_timestamp in frames_list:
                            diff = abs(frame_timestamp - timestamp_seconds)
                            if diff < min_diff:
                                min_diff = diff
                                closest_frame = frame_path
                        
                        if closest_frame and os.path.exists(closest_frame):
                            with cols[idx % len(cols)]:
                                try:
                                    img = Image.open(closest_frame)
                                    st.image(
                                        img,
                                        caption=f"At {timestamp_str} ({timestamp_seconds:.1f}s)",
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.warning(f"Could not load frame: {e}")
                
                # Show context in expander
                with st.expander("🔍 View Retrieved Context"):
                    st.text(context)
            
            # Update chat history
            st.session_state.chat_history[-1] = ("user", user_query, answer, context)
            st.session_state.chat_history.append(("assistant", "", answer, context))
        
        else:
            st.error("Failed to get answer. Please try again.")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("👈 Please upload and process a video file using the sidebar to get started.")
    
    # Instructions
    with st.expander("📖 How to use VideoQuery"):
        st.markdown("""
        **VideoQuery** is a Multimodal RAG application that allows you to:
        
        1. **Upload a video** - Upload an MP4 video file
        2. **Process the video** - The system will:
           - Extract audio and transcribe it using Whisper
           - Extract frames every 5 seconds
           - Generate embeddings for frames using CLIP
           - Store everything in ChromaDB
        3. **Ask questions** - Query the video content using natural language
        4. **Get answers** - Receive AI-generated answers with relevant frames
        
        **Features:**
        - Visual search: Find frames based on visual content
        - Text search: Find relevant transcript segments
        - Timestamp citations: See frames at specific timestamps mentioned in answers
        """)

# Footer
st.markdown("---")
st.markdown("**VideoQuery** - Multimodal RAG for Video Content")

