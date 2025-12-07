"""
Video Ingestion Pipeline for VideoQuery
Handles video processing, audio extraction, transcription, and frame embeddings.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ffmpeg
import whisper
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class VideoProcessor:
    """
    Processes video files to extract audio, frames, transcriptions, and embeddings.
    """
    
    def __init__(self, whisper_model_size: str = "small", device: Optional[str] = None):
        """
        Initialize the VideoProcessor.
        
        Args:
            whisper_model_size: Size of Whisper model to use (tiny, base, small, medium, large)
            device: Device to run models on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model_size = whisper_model_size
        
        # Initialize Whisper model
        print(f"Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
        
        # Initialize CLIP model
        print(f"Loading CLIP model on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("VideoProcessor initialized successfully!")
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to the input video file
            output_path: Optional path for output audio file. If None, creates temp file.
        
        Returns:
            Path to the extracted audio file
        
        Raises:
            Exception: If ffmpeg extraction fails
        """
        try:
            if output_path is None:
                # Create temporary audio file
                temp_dir = tempfile.gettempdir()
                video_name = Path(video_path).stem
                output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            # Extract audio using ffmpeg
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            print(f"Audio extracted successfully to: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_message = f"FFmpeg error during audio extraction: {e.stderr.decode() if e.stderr else str(e)}"
            raise Exception(error_message) from e
        except Exception as e:
            raise Exception(f"Error extracting audio: {str(e)}") from e
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None, 
                      interval: int = 5) -> List[Tuple[str, float]]:
        """
        Extract frames from video at specified intervals using ffmpeg.
        
        Args:
            video_path: Path to the input video file
            output_dir: Optional directory for output frames. If None, creates temp directory.
            interval: Extract 1 frame every N seconds (default: 5)
        
        Returns:
            List of tuples (frame_path, timestamp_in_seconds)
        
        Raises:
            Exception: If ffmpeg extraction fails
        """
        try:
            if output_dir is None:
                # Create temporary directory for frames
                temp_dir = tempfile.mkdtemp()
                output_dir = temp_dir
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            video_name = Path(video_path).stem
            frame_pattern = os.path.join(output_dir, f"{video_name}_frame_%04d.png")
            
            # Extract frames at specified interval using ffmpeg
            # -vf "fps=1/5" means 1 frame per 5 seconds
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream, 
                frame_pattern,
                vf=f"fps=1/{interval}",
                vsync=0
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Get list of extracted frames with timestamps
            frame_files = sorted(Path(output_dir).glob(f"{video_name}_frame_*.png"))
            frames_with_timestamps = []
            
            for frame_file in frame_files:
                # Extract frame number from filename
                frame_num = int(frame_file.stem.split('_')[-1])
                timestamp = frame_num * interval
                frames_with_timestamps.append((str(frame_file), timestamp))
            
            print(f"Extracted {len(frames_with_timestamps)} frames from video")
            return frames_with_timestamps
            
        except ffmpeg.Error as e:
            error_message = f"FFmpeg error during frame extraction: {e.stderr.decode() if e.stderr else str(e)}"
            raise Exception(error_message) from e
        except Exception as e:
            raise Exception(f"Error extracting frames: {str(e)}") from e
    
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, any]]:
        """
        Transcribe audio using Whisper model with timestamps.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            List of dictionaries with 'text', 'start', and 'end' keys for each segment
        
        Raises:
            Exception: If transcription fails
        """
        try:
            print(f"Transcribing audio: {audio_path}")
            result = self.whisper_model.transcribe(
                audio_path,
                verbose=False,
                word_timestamps=False
            )
            
            # Format segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "text": segment.get("text", "").strip(),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0)
                })
            
            print(f"Transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}") from e
    
    def get_frame_embeddings(self, frame_paths: List[str]) -> np.ndarray:
        """
        Convert frames to vector embeddings using CLIP.
        
        Args:
            frame_paths: List of paths to frame images
        
        Returns:
            numpy array of shape (num_frames, embedding_dim) with normalized embeddings
        
        Raises:
            Exception: If embedding generation fails
        """
        try:
            if not frame_paths:
                return np.array([])
            
            print(f"Generating embeddings for {len(frame_paths)} frames...")
            
            # Load and process images
            images = []
            for frame_path in frame_paths:
                try:
                    image = Image.open(frame_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Warning: Could not load frame {frame_path}: {e}")
                    continue
            
            if not images:
                raise Exception("No valid frames to process")
            
            # Process images with CLIP
            inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy and move to CPU
            embeddings = image_features.cpu().numpy()
            
            print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error generating frame embeddings: {str(e)}") from e
    
    def process_video(self, video_path: str, output_dir: Optional[str] = None,
                     frame_interval: int = 5) -> Dict[str, any]:
        """
        Complete video processing pipeline: extract audio, frames, transcribe, and embed.
        
        Args:
            video_path: Path to the input video file
            output_dir: Optional directory for intermediate files
            frame_interval: Extract 1 frame every N seconds (default: 5)
        
        Returns:
            Dictionary containing:
                - audio_path: Path to extracted audio
                - frames: List of (frame_path, timestamp) tuples
                - transcript: List of transcription segments with timestamps
                - frame_embeddings: numpy array of frame embeddings
        """
        try:
            print(f"Processing video: {video_path}")
            
            # Extract audio - construct proper audio file path if output_dir is provided
            audio_path = None
            if output_dir:
                video_name = Path(video_path).stem
                audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")
            audio_path = self.extract_audio(video_path, audio_path)
            
            # Extract frames
            frames = self.extract_frames(video_path, output_dir, frame_interval)
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            
            # Generate frame embeddings
            frame_paths = [frame_path for frame_path, _ in frames]
            frame_embeddings = self.get_frame_embeddings(frame_paths)
            
            result = {
                "audio_path": audio_path,
                "frames": frames,
                "transcript": transcript,
                "frame_embeddings": frame_embeddings
            }
            
            print("Video processing completed successfully!")
            return result
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}") from e

