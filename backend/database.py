"""
ChromaDB Database Management for VideoQuery
Handles persistent storage of video frames and transcripts.
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import uuid


class VideoDatabase:
    """
    Manages ChromaDB collections for storing video frames and transcripts.
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize ChromaDB client with persistent storage.
        
        Args:
            persist_directory: Directory where ChromaDB will persist data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections
        self.video_frames_collection = self._get_or_create_collection("video_frames")
        self.video_transcripts_collection = self._get_or_create_collection("video_transcripts")
        
        print(f"ChromaDB initialized at: {persist_directory}")
        print(f"Collections ready: video_frames, video_transcripts")
    
    def _get_or_create_collection(self, collection_name: str):
        """
        Get existing collection or create a new one.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            ChromaDB collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": f"Collection for {collection_name}"}
            )
            print(f"Created new collection: {collection_name}")
        
        return collection
    
    def store_video_data(self, video_id: str, processor_output: Dict) -> Dict[str, int]:
        """
        Store video data (frames and transcripts) in ChromaDB.
        
        Args:
            video_id: Unique identifier for the video
            processor_output: Output dictionary from VideoProcessor.process_video() containing:
                - frames: List of (frame_path, timestamp) tuples
                - transcript: List of dicts with 'text', 'start', 'end' keys
                - frame_embeddings: numpy array of frame embeddings
        
        Returns:
            Dictionary with counts of stored items:
                - frames_stored: Number of frames stored
                - transcripts_stored: Number of transcript segments stored
        """
        frames_stored = 0
        transcripts_stored = 0
        
        # Store frame embeddings
        frames = processor_output.get("frames", [])
        frame_embeddings = processor_output.get("frame_embeddings", [])
        
        if frames and len(frame_embeddings) > 0:
            frames_stored = self._store_frames(
                video_id=video_id,
                frames=frames,
                embeddings=frame_embeddings
            )
        
        # Store transcript segments
        transcript = processor_output.get("transcript", [])
        if transcript:
            transcripts_stored = self._store_transcripts(
                video_id=video_id,
                transcript=transcript
            )
        
        print(f"Stored video data for video_id '{video_id}': "
              f"{frames_stored} frames, {transcripts_stored} transcript segments")
        
        return {
            "frames_stored": frames_stored,
            "transcripts_stored": transcripts_stored
        }
    
    def _store_frames(self, video_id: str, frames: List[tuple], embeddings) -> int:
        """
        Store frame embeddings in the video_frames collection.
        
        Args:
            video_id: Unique identifier for the video
            frames: List of (frame_path, timestamp) tuples
            embeddings: numpy array of frame embeddings
        
        Returns:
            Number of frames stored
        """
        if len(frames) != len(embeddings):
            raise ValueError(
                f"Mismatch between number of frames ({len(frames)}) "
                f"and embeddings ({len(embeddings)})"
            )
        
        ids = []
        embeddings_list = []
        metadatas = []
        
        for i, (frame_path, timestamp) in enumerate(frames):
            # Generate unique ID for this frame
            frame_id = f"{video_id}_frame_{i}_{uuid.uuid4().hex[:8]}"
            ids.append(frame_id)
            
            # Convert numpy array to list for ChromaDB
            embeddings_list.append(embeddings[i].tolist())
            
            # Store metadata with timestamp and video_id
            metadatas.append({
                "timestamp": float(timestamp),
                "video_id": video_id,
                "frame_path": frame_path,
                "frame_index": i
            })
        
        # Add to collection
        self.video_frames_collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        return len(ids)
    
    def _store_transcripts(self, video_id: str, transcript: List[Dict]) -> int:
        """
        Store transcript segments in the video_transcripts collection.
        
        Args:
            video_id: Unique identifier for the video
            transcript: List of dicts with 'text', 'start', 'end' keys
        
        Returns:
            Number of transcript segments stored
        """
        ids = []
        texts = []
        metadatas = []
        
        for i, segment in enumerate(transcript):
            # Use sequential chunk IDs: chunk_0, chunk_1, chunk_2, etc.
            # Format: {video_id}_chunk_{i} to ensure uniqueness across videos
            # The window retrieval will parse this to extract the index
            segment_id = f"{video_id}_chunk_{i}"
            ids.append(segment_id)
            
            # Store the text (ChromaDB will auto-embed this)
            texts.append(segment.get("text", ""))
            
            # Store metadata with timestamps and video_id
            metadatas.append({
                "timestamp_start": float(segment.get("start", 0.0)),
                "timestamp_end": float(segment.get("end", 0.0)),
                "video_id": video_id,
                "segment_index": i,
                "chunk_index": i  # Store index for easy retrieval
            })
        
        # Add to collection
        self.video_transcripts_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return len(ids)
    
    def query_frames(self, query_embedding: List[float], n_results: int = 5,
                    video_id_filter: Optional[str] = None) -> Dict:
        """
        Query frame embeddings by similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            video_id_filter: Optional video_id to filter results
        
        Returns:
            Dictionary with query results
        """
        where = {"video_id": video_id_filter} if video_id_filter else None
        
        results = self.video_frames_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def query_transcripts(self, query_text: str, n_results: int = 5,
                         video_id_filter: Optional[str] = None) -> Dict:
        """
        Query transcript segments by text similarity.
        
        Args:
            query_text: Query text string
            n_results: Number of results to return
            video_id_filter: Optional video_id to filter results
        
        Returns:
            Dictionary with query results
        """
        where = {"video_id": video_id_filter} if video_id_filter else None
        
        results = self.video_transcripts_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_video_frames(self, video_id: str) -> Dict:
        """
        Get all frames for a specific video.
        
        Args:
            video_id: Unique identifier for the video
        
        Returns:
            Dictionary with all frames for the video
        """
        results = self.video_frames_collection.get(
            where={"video_id": video_id}
        )
        return results
    
    def get_video_transcripts(self, video_id: str) -> Dict:
        """
        Get all transcript segments for a specific video.
        
        Args:
            video_id: Unique identifier for the video
        
        Returns:
            Dictionary with all transcript segments for the video
        """
        results = self.video_transcripts_collection.get(
            where={"video_id": video_id}
        )
        return results


def initialize_database(persist_directory: str = "./data/chromadb") -> VideoDatabase:
    """
    Initialize and return a VideoDatabase instance.
    
    Args:
        persist_directory: Directory where ChromaDB will persist data
    
    Returns:
        VideoDatabase instance
    """
    return VideoDatabase(persist_directory=persist_directory)

