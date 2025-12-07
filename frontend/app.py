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
if "transcript_ready" not in st.session_state:
    st.session_state.transcript_ready = False
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""


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


def extract_timestamp_citations(text: str) -> list:
    """
    Extract timestamp citations from text in format [[seconds]].
    
    Returns:
        List of timestamp values in seconds (float)
    """
    # Pattern for [[12.5]] or [[24]] format - captures float numbers cleanly
    pattern = r"\[\[(\d+(?:\.\d+)?)\]\]"
    matches = re.findall(pattern, text)
    # Convert to float and return
    timestamps = [float(match) for match in matches]
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


def get_video_transcript(video_id: str):
    """Get full transcript for a video."""
    try:
        response = requests.get(f"{API_BASE_URL}/video/{video_id}/transcript")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting transcript: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Please make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_video_summary(video_id: str):
    """Get video summary."""
    try:
        response = requests.get(f"{API_BASE_URL}/summary/{video_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error generating summary: {response.json().get('detail', 'Unknown error')}")
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
    
    # Download Transcript button in sidebar
    if st.session_state.video_id:
        st.sidebar.markdown("---")
        st.sidebar.header("📄 Transcript")
        
        if st.sidebar.button("📥 Download Transcript", use_container_width=True):
            with st.spinner("Retrieving transcript..."):
                transcript_data = get_video_transcript(st.session_state.video_id)
                
                if transcript_data and transcript_data.get("documents"):
                    # Sort transcript segments by timestamp
                    documents = transcript_data.get("documents", [])
                    metadatas = transcript_data.get("metadatas", [])
                    
                    transcript_parts = []
                    for doc, metadata in zip(documents, metadatas):
                        start_time = metadata.get("timestamp_start", 0.0)
                        transcript_parts.append((start_time, doc))
                    
                    transcript_parts.sort(key=lambda x: x[0])
                    
                    # Format transcript with timestamps
                    transcript_text = "Video Transcript\n"
                    transcript_text += "=" * 50 + "\n\n"
                    
                    for start_time, doc in transcript_parts:
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        timestamp_str = f"{minutes:02d}:{seconds:02d}"
                        transcript_text += f"[{timestamp_str}] {doc}\n\n"
                    
                    # Store transcript in session state for download
                    st.session_state.transcript_text = transcript_text
                    st.session_state.transcript_ready = True
                else:
                    st.sidebar.error("No transcript available for this video.")
                    st.session_state.transcript_ready = False
        
        # Show download button if transcript is ready
        if st.session_state.get("transcript_ready", False):
            st.sidebar.download_button(
                label="⬇️ Download as .txt",
                data=st.session_state.transcript_text,
                file_name=f"transcript_{st.session_state.video_id}.txt",
                mime="text/plain",
                use_container_width=True
            )

# Main content area
if st.session_state.video_id:
    st.success(f"✅ Video loaded (ID: {st.session_state.video_id})")
    
    # Generate Summary section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📝 Video Summary")
    with col2:
        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Generating summary... This may take a moment."):
                summary_data = get_video_summary(st.session_state.video_id)
                
                if summary_data:
                    st.session_state.video_summary = summary_data.get("summary", "")
    
    # Display summary if available
    if "video_summary" in st.session_state and st.session_state.video_summary:
        st.markdown("---")
        # Create a nice card-style display for the summary
        with st.container():
            st.markdown("### 📋 Key Takeaways")
            # Use a styled info box for the summary
            st.markdown(
                f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; margin: 10px 0;">'
                f'<p style="font-size: 16px; line-height: 1.8; color: #1f2937; margin: 0;">{st.session_state.video_summary}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("---")
    
    # Chat interface
    st.header("💬 Ask Questions About the Video")
    
    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Show chat history - simplified to show only questions and answers
    for i, (role, message, answer, context) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            if role == "user":
                # Show user question
                st.write(message)
            elif role == "assistant" and answer:
                # Show AI answer
                st.write(answer)
                
                # Extract timestamp citations in [[seconds]] format
                timestamp_citations = extract_timestamp_citations(answer)
                
                # Also extract traditional timestamps (MM:SS format) for backward compatibility
                traditional_timestamps = extract_timestamps(answer)
                traditional_seconds = [timestamp_to_seconds(ts) for ts in traditional_timestamps]
                
                # Combine both types of timestamps
                all_timestamps_seconds = list(set(timestamp_citations + traditional_seconds))
                
                if all_timestamps_seconds and st.session_state.frames_data.get("frames"):
                    # Get frames from session state
                    frames_list = st.session_state.frames_data.get("frames", [])
                    
                    # Display frames for found timestamps immediately below answer
                    cols = st.columns(min(len(all_timestamps_seconds), 3))
                    
                    for idx, timestamp_seconds in enumerate(all_timestamps_seconds):
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
                                    # Format caption: "Frame at 24s" or "Frame at 24.5s"
                                    caption = f"Frame at {int(timestamp_seconds)}s" if timestamp_seconds.is_integer() else f"Frame at {timestamp_seconds:.1f}s"
                                    st.image(
                                        img,
                                        caption=caption,
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.warning(f"Could not load frame: {e}")
                
                # Hide debug context in expander
                if context:
                    with st.expander("🔍 View Debug Context"):
                        st.text(context)
    
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
                # Show AI answer only
                st.write(answer)
                
                # Extract timestamp citations in [[seconds]] format
                timestamp_citations = extract_timestamp_citations(answer)
                
                # Also extract traditional timestamps (MM:SS format) for backward compatibility
                traditional_timestamps = extract_timestamps(answer)
                traditional_seconds = [timestamp_to_seconds(ts) for ts in traditional_timestamps]
                
                # Combine both types of timestamps
                all_timestamps_seconds = list(set(timestamp_citations + traditional_seconds))
                
                if all_timestamps_seconds and st.session_state.frames_data.get("frames"):
                    # Get frames from session state
                    frames_list = st.session_state.frames_data.get("frames", [])
                    
                    # Display frames for found timestamps immediately below answer
                    cols = st.columns(min(len(all_timestamps_seconds), 3))
                    
                    for idx, timestamp_seconds in enumerate(all_timestamps_seconds):
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
                                    # Format caption: "Frame at 24s" or "Frame at 24.5s"
                                    caption = f"Frame at {int(timestamp_seconds)}s" if timestamp_seconds.is_integer() else f"Frame at {timestamp_seconds:.1f}s"
                                    st.image(
                                        img,
                                        caption=caption,
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.warning(f"Could not load frame: {e}")
                
                # Hide debug context in expander
                if context:
                    with st.expander("🔍 View Debug Context"):
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

