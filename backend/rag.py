"""
RAG (Retrieval Augmented Generation) Module for VideoQuery
Handles querying video content and generating answers using LLM.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
from backend.database import VideoDatabase


class VideoRAG:
    """
    Retrieval Augmented Generation system for video queries.
    """
    
    def __init__(self, database: VideoDatabase, device: Optional[str] = None):
        """
        Initialize the VideoRAG system.
        
        Args:
            database: VideoDatabase instance
            device: Device to run models on ('cuda', 'cpu', or None for auto-detection)
        """
        self.database = database
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CLIP model for text encoding
        print(f"Loading CLIP model for text encoding on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it with: export GROQ_API_KEY='your-api-key'"
            )
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = "llama3-8b-8192"
        
        print("VideoRAG initialized successfully!")
    
    def encode_text_query(self, text: str) -> np.ndarray:
        """
        Encode text query to embedding using CLIP.
        
        Args:
            text: Text query string
        
        Returns:
            Normalized text embedding as numpy array
        """
        # Process text with CLIP
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text embedding
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            # Normalize embedding
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Convert to numpy and move to CPU
        embedding = text_features.cpu().numpy()[0]
        
        return embedding
    
    def query_video(self, user_query: str, video_id_filter: Optional[str] = None,
                   top_k_frames: int = 3, top_k_transcripts: int = 3) -> Dict:
        """
        Query video content and generate answer using RAG.
        
        Args:
            user_query: User's question about the video
            video_id_filter: Optional video_id to filter search results
            top_k_frames: Number of top matching frames to retrieve
            top_k_transcripts: Number of top matching transcript segments to retrieve
        
        Returns:
            Dictionary containing:
                - answer: Generated answer from LLM
                - frame_results: Retrieved frame matches
                - transcript_results: Retrieved transcript matches
                - context: Combined context string
        """
        print(f"Processing query: {user_query}")
        
        # Step 1: Encode user query to text embedding using CLIP
        print("Encoding query to embedding...")
        query_embedding = self.encode_text_query(user_query)
        query_embedding_list = query_embedding.tolist()
        
        # Step 2: Search video_frames collection for top matching images
        print(f"Searching video_frames collection for top {top_k_frames} matches...")
        frame_results = self.database.query_frames(
            query_embedding=query_embedding_list,
            n_results=top_k_frames,
            video_id_filter=video_id_filter
        )
        
        # Step 3: Search video_transcripts collection for top matching text segments
        print(f"Searching video_transcripts collection for top {top_k_transcripts} matches...")
        transcript_results = self.database.query_transcripts(
            query_text=user_query,
            n_results=top_k_transcripts,
            video_id_filter=video_id_filter
        )
        
        # Step 4: Combine results into context string
        context = self._build_context(frame_results, transcript_results)
        
        # Step 5: Send context + user query to Llama 3 (placeholder)
        print("Generating answer with LLM...")
        answer = self._generate_answer(user_query, context)
        
        result = {
            "answer": answer,
            "frame_results": frame_results,
            "transcript_results": transcript_results,
            "context": context
        }
        
        print("Query processing completed!")
        return result
    
    def _build_context(self, frame_results: Dict, transcript_results: Dict) -> str:
        """
        Build context string from retrieved frames and transcripts.
        
        Args:
            frame_results: ChromaDB query results for frames
            transcript_results: ChromaDB query results for transcripts
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add frame information
        if frame_results and frame_results.get("ids") and len(frame_results["ids"][0]) > 0:
            context_parts.append("=== Relevant Video Frame Descriptions ===")
            frame_ids = frame_results["ids"][0]
            frame_metadatas = frame_results["metadatas"][0]
            frame_distances = frame_results.get("distances", [[]])[0] if "distances" in frame_results else [0.0] * len(frame_ids)
            
            for i, (frame_id, metadata, distance) in enumerate(zip(frame_ids, frame_metadatas, frame_distances), 1):
                timestamp = metadata.get("timestamp", 0.0)
                video_id = metadata.get("video_id", "unknown")
                context_parts.append(
                    f"Frame {i}: At {timestamp:.2f} seconds (video: {video_id}, "
                    f"similarity: {1-distance:.3f})"
                )
        
        # Add transcript information
        if transcript_results and transcript_results.get("ids") and len(transcript_results["ids"][0]) > 0:
            context_parts.append("\n=== Relevant Transcript Segments ===")
            transcript_ids = transcript_results["ids"][0]
            transcript_documents = transcript_results["documents"][0]
            transcript_metadatas = transcript_results["metadatas"][0]
            transcript_distances = transcript_results.get("distances", [[]])[0] if "distances" in transcript_results else [0.0] * len(transcript_ids)
            
            for i, (transcript_id, document, metadata, distance) in enumerate(
                zip(transcript_ids, transcript_documents, transcript_metadatas, transcript_distances), 1
            ):
                start_time = metadata.get("timestamp_start", 0.0)
                end_time = metadata.get("timestamp_end", 0.0)
                video_id = metadata.get("video_id", "unknown")
                context_parts.append(
                    f"Segment {i} ({start_time:.2f}s - {end_time:.2f}s, video: {video_id}, "
                    f"similarity: {1-distance:.3f}):\n{document}"
                )
        
        context = "\n".join(context_parts) if context_parts else "No relevant content found."
        return context
    
    def _generate_answer(self, user_query: str, context: str) -> str:
        """
        Generate answer using Llama 3 via Groq API.
        
        Args:
            user_query: User's question
            context: Retrieved context from video (includes transcript segments and frame descriptions)
        
        Returns:
            Generated answer string
        """
        # System prompt
        system_prompt = (
            "You are a helpful video assistant. Answer the user question based ONLY on the "
            "provided video context. If the answer is not in the context, say you don't know."
        )
        
        # Construct user prompt with context
        user_prompt = f"""Video Context:
{context}

User Question: {user_query}

Please provide a helpful answer based on the video context above."""
        
        try:
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            error_msg = f"Error calling Groq API: {str(e)}"
            print(error_msg)
            raise Exception(error_msg) from e


def query_video(user_query: str, database: VideoDatabase, 
                video_id_filter: Optional[str] = None) -> Dict:
    """
    Convenience function to query video content.
    
    Args:
        user_query: User's question about the video
        database: VideoDatabase instance
        video_id_filter: Optional video_id to filter search results
    
    Returns:
        Dictionary with answer and retrieval results
    """
    rag = VideoRAG(database=database)
    return rag.query_video(user_query, video_id_filter=video_id_filter)

