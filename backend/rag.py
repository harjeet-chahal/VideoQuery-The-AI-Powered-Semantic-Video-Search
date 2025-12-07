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
        self.groq_model = "llama-3.1-8b-instant"
        
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
    
    def get_context_window(self, results: Dict, video_id: Optional[str] = None, window_size: int = 2) -> str:
        """
        For every 'hit' in the database, fetch the neighbors (chunks before and after).
        This fixes the issue where we find the question but miss the answer.
        
        Args:
            results: ChromaDB query results containing IDs and documents
            video_id: Video ID to filter chunks (required for proper windowing)
            window_size: Number of chunks before and after to include
        
        Returns:
            Expanded context text with neighboring chunks
        """
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return ""
        
        expanded_context = []
        collection = self.database.video_transcripts_collection
        
        # Get all chunks for this video to determine valid index range
        if video_id:
            all_chunks = collection.get(where={"video_id": video_id})
            if not all_chunks or not all_chunks.get("ids"):
                return ""
            # Find max chunk index for this video
            max_index = -1
            for chunk_id in all_chunks["ids"]:
                if chunk_id.startswith(f"{video_id}_chunk_"):
                    try:
                        idx = int(chunk_id.split("_")[-1])
                        max_index = max(max_index, idx)
                    except (ValueError, IndexError):
                        continue
        else:
            max_index = 10000  # Fallback if no video_id
        
        # Process each match
        for match_id in results["ids"][0]:
            # Parse chunk index from ID format: {video_id}_chunk_{index}
            try:
                parts = match_id.split("_")
                if len(parts) >= 3 and parts[-2] == "chunk":
                    current_idx = int(parts[-1])
                else:
                    # Fallback: try to extract number from end
                    current_idx = int(match_id.split("_")[-1])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse chunk index from ID: {match_id}")
                continue
            
            # Calculate neighbor IDs
            start_idx = max(0, current_idx - window_size)
            end_idx = min(max_index + 1, current_idx + window_size + 1)
            
            # Create list of neighbor IDs to fetch
            neighbor_ids = [f"{video_id}_chunk_{n}" for n in range(start_idx, end_idx)]
            
            # Fetch these neighbors from DB
            try:
                neighbors = collection.get(ids=neighbor_ids)
                
                if neighbors and neighbors.get("ids") and neighbors.get("documents") and neighbors.get("metadatas"):
                    # Sort them by ID to ensure they read like a normal paragraph
                    sorted_data = sorted(
                        zip(neighbors["ids"], neighbors["documents"], neighbors["metadatas"]),
                        key=lambda x: int(x[0].split("_")[-1]) if x[0].split("_")[-1].isdigit() else 0
                    )
                    
                    # Join them into a single coherent block of text with timestamps
                    block_parts = []
                    for chunk_id, doc, metadata in sorted_data:
                        # Get timestamp from metadata (use start time)
                        timestamp_start = metadata.get("timestamp_start", 0.0)
                        # Format: [Time: 12.5s] text content
                        block_parts.append(f"[Time: {timestamp_start:.1f}s] {doc}")
                    
                    block_text = " ".join(block_parts)
                    expanded_context.append(block_text)
            except Exception as e:
                print(f"Warning: Error fetching neighbors for {match_id}: {e}")
                continue
        
        # Remove duplicates and join
        unique_contexts = list(set(expanded_context))
        return "\n\n---\n\n".join(unique_contexts)
    
    def query_video(self, user_query: str, video_id_filter: Optional[str] = None,
                   top_k_frames: int = 3, top_k_transcripts: int = 5) -> Dict:
        """
        Query video content and generate answer using RAG with Window Retrieval.
        
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
        
        # Step 4: Expand context using Window Retrieval
        print("Expanding context with window retrieval...")
        context_text = self.get_context_window(
            transcript_results,
            video_id=video_id_filter,
            window_size=2
        )
        
        # Step 5: Build frame context (keep original format for frames)
        frame_context = ""
        if frame_results and frame_results.get("ids") and len(frame_results["ids"][0]) > 0:
            frame_context = "=== Relevant Video Frame Descriptions ===\n"
            frame_ids = frame_results["ids"][0]
            frame_metadatas = frame_results["metadatas"][0]
            for i, (frame_id, metadata) in enumerate(zip(frame_ids, frame_metadatas), 1):
                timestamp = metadata.get("timestamp", 0.0)
                frame_context += f"Frame {i}: At {timestamp:.2f} seconds\n"
        
        # Combine contexts
        if frame_context:
            context = f"{frame_context}\n\n=== Transcript Context ===\n{context_text}"
        else:
            context = context_text if context_text else "No relevant content found."
        
        # Step 6: Generate answer with improved prompt
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
    
    def _generate_answer(self, user_query: str, context: str) -> str:
        """
        Generate answer using Llama 3 via Groq API with improved system prompt.
        
        Args:
            user_query: User's question
            context: Retrieved context from video (includes transcript segments and frame descriptions)
        
        Returns:
            Generated answer string
        """
        # Improved system prompt for better answer extraction
        system_prompt = """You are a video analyst. Answer the user's question based on the transcript segments.

CITATION RULE: You MUST cite the timestamp for every claim using EXACTLY this format: [[123]] (just the seconds number, no 's', no 'Time:', double brackets).

Example: Correct: "LangChain uses agents [[341.9]]."
Incorrect: "LangChain uses agents [Time: 341.9s]."

INSTRUCTIONS:
1. The segments may be out of order or contain the speaker asking rhetorical questions. 
2. Look for the *answer* in the text, not just the question.
3. If the text mentions "X is..." or "X refers to...", that is your definition.
4. Every claim you make MUST include a timestamp citation in [[seconds]] format.
5. If the answer is not in the context, say you don't know."""
        
        # Construct user prompt with context
        user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
        
        try:
            # Call Groq API with lower temperature for more factual answers
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual answers
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

