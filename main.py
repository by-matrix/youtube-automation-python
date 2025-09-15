# YouTube Automation Microservices
# Optimized for Render deployment

import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import requests
import json
from datetime import datetime, timedelta
import random
import tempfile
import subprocess
import shutil
from pathlib import Path

app = FastAPI(
    title="YouTube Automation FREE", 
    version="1.0.0",
    description="Free YouTube to social media automation"
)

# Enable CORS for n8n integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RENDER_DISK_PATH = "/opt/render/project/src"

# Create necessary directories
os.makedirs("downloads", exist_ok=True)
os.makedirs("clips", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Data models
class VideoRequest(BaseModel):
    youtube_url: str
    shorts_count: int = 3
    chat_id: str = ""

class AnalyzeRequest(BaseModel):
    youtube_url: str

class OptimizeRequest(BaseModel):
    video_title: str
    video_description: str
    clip_transcript: str = ""
    niche: str = "general"

# === HEALTH CHECK ===
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "YouTube Automation FREE",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/analyze-video",
            "/download-video", 
            "/generate-clips",
            "/optimize-content",
            "/health"
        ]
    }

@app.get("/health")
async def detailed_health():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "disk_space": shutil.disk_usage("/")._asdict(),
        "temp_files": len(os.listdir("temp")) if os.path.exists("temp") else 0,
        "downloads": len(os.listdir("downloads")) if os.path.exists("downloads") else 0,
        "clips": len(os.listdir("clips")) if os.path.exists("clips") else 0
    }

# === VIDEO ANALYSIS SERVICE ===
@app.post("/api/analyze-video")
async def analyze_video(request: AnalyzeRequest):
    """Analyze YouTube video and suggest optimal clip count"""
    try:
        # Configure yt-dlp for info extraction only (no download)
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.youtube_url, download=False)
            
            duration = info.get('duration', 0)
            title = info.get('title', 'Unknown Video')
            view_count = info.get('view_count', 0)
            description = info.get('description', '')[:200] + "..." if info.get('description') else ""
            
            # Smart clip count suggestions
            if duration < 120:  # 2 minutes
                suggested_clips = 2
                reason = "Short video - 2 clips recommended"
            elif duration < 600:  # 10 minutes
                suggested_clips = 5
                reason = "Medium video - 5 clips for best coverage"
            elif duration < 1800:  # 30 minutes
                suggested_clips = 8
                reason = "Long video - 8 clips to capture highlights"
            else:
                suggested_clips = 10
                reason = "Very long video - maximum clips recommended"
            
            analysis = {
                "status": "success",
                "video_id": info['id'],
                "title": title,
                "description": description,
                "duration": duration,
                "duration_formatted": f"{duration//60}:{duration%60:02d}",
                "view_count": view_count,
                "suggested_clips": suggested_clips,
                "reason": reason,
                "thumbnail": info.get('thumbnail', ''),
                "uploader": info.get('uploader', 'Unknown')
            }
            
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# === VIDEO PROCESSING SERVICE ===
@app.post("/api/download-video")
async def download_video(request: VideoRequest):
    """Download YouTube video for processing"""
    try:
        # Clean old files first
        cleanup_old_files()
        
        # Configure yt-dlp for download
        output_path = f"downloads/%(id)s.%(ext)s"
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.youtube_url, download=True)
            
            video_data = {
                "status": "success",
                "video_id": info['id'],
                "title": info['title'],
                "description": info.get('description', ''),
                "duration": info['duration'],
                "filepath": f"downloads/{info['id']}.mp4",
                "shorts_count": request.shorts_count,
                "chat_id": request.chat_id
            }
            
        return video_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# === CLIP GENERATION SERVICE ===
@app.post("/api/generate-clips")
async def generate_clips(video_data: dict):
    """Generate short clips using FFmpeg"""
    try:
        video_id = video_data['video_id']
        shorts_count = video_data['shorts_count']
        duration = video_data['duration']
        
        video_path = f"downloads/{video_id}.mp4"
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        clips = []
        clip_duration = 30  # 30 seconds per clip
        
        # Calculate optimal start times
        safe_start = min(30, duration * 0.1)  # Skip first 10% or 30 seconds
        safe_end = min(30, duration * 0.1)    # Skip last 10% or 30 seconds
        usable_duration = duration - safe_start - safe_end
        
        if usable_duration < clip_duration * shorts_count:
            # Video too short, use simple intervals
            interval = max(clip_duration, duration / (shorts_count + 1))
        else:
            interval = usable_duration / shorts_count
        
        for i in range(shorts_count):
            start_time = safe_start + (i * interval)
            
            # Ensure we don't exceed video duration
            if start_time + clip_duration > duration - safe_end:
                start_time = duration - safe_end - clip_duration
            
            output_path = f"clips/{video_id}_clip_{i+1}.mp4"
            
            # Generate clip using FFmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(clip_duration),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                '-c:v', 'libx264', '-c:a', 'aac',
                '-y', output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                clips.append({
                    "clip_id": f"{video_id}_clip_{i+1}",
                    "filepath": output_path,
                    "start_time": start_time,
                    "duration": clip_duration,
                    "start_formatted": f"{int(start_time//60)}:{int(start_time%60):02d}"
                })
            else:
                print(f"FFmpeg error for clip {i+1}: {result.stderr}")
                
        return {"status": "success", "clips": clips, **video_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clip generation failed: {str(e)}")

# === GEMINI AI OPTIMIZATION SERVICE ===
@app.post("/api/optimize-content")
async def optimize_content(request: OptimizeRequest):
    """Optimize content using Gemini AI PRO"""
    try:
        if not GEMINI_API_KEY:
            # Fallback optimization without AI
            return create_fallback_optimization(request)
        
        # Call Gemini AI API
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Create viral social media content for this video:
        
        Title: {request.video_title}
        Description: {request.video_description[:500]}
        Niche: {request.niche}
        
        Generate:
        1. Viral YouTube title (50-60 characters)
        2. YouTube description (2-3 engaging lines)
        3. Instagram caption with emojis (100-150 words)
        4. Twitter/X post (250 characters max)
        5. Pinterest description (80 words)
        6. 8 trending hashtags for each platform
        
        Make it attention-grabbing, engaging, and platform-optimized!
        Focus on hooks, curiosity, and shareability.
        """
        
        response = model.generate_content(prompt)
        ai_content = response.text
        
        # Parse AI response (simplified parsing)
        optimization = {
            "status": "success",
            "ai_generated": True,
            "viral_title": extract_content(ai_content, "title") or f"🔥 {request.video_title}",
            "youtube_description": extract_content(ai_content, "description") or "Amazing content that will blow your mind! 🤯",
            "instagram_caption": extract_content(ai_content, "instagram") or f"🔥 Check out this amazing content! {request.video_title} #viral #trending",
            "twitter_post": extract_content(ai_content, "twitter") or f"Mind-blowing content! 🤯 {request.video_title} #viral",
            "pinterest_description": extract_content(ai_content, "pinterest") or f"Discover amazing content: {request.video_title}",
            "hashtags": {
                "youtube": ["#shorts", "#viral", "#trending", "#amazing"],
                "instagram": ["#reels", "#viral", "#trending", "#content"],
                "twitter": ["#viral", "#trending", "#content"],
                "pinterest": ["#viral", "#amazing", "#content"]
            },
            "full_ai_response": ai_content
        }
        
        return optimization
        
    except Exception as e:
        print(f"AI optimization error: {e}")
        return create_fallback_optimization(request)

def create_fallback_optimization(request: OptimizeRequest):
    """Create optimization without AI as fallback"""
    title = request.video_title
    return {
        "status": "success",
        "ai_generated": False,
        "viral_title": f"🔥 {title[:50]}",
        "youtube_description": f"Amazing content! Watch this incredible video about {title}. Like and subscribe for more!",
        "instagram_caption": f"🔥 {title}\n\nWhat do you think about this? Comment below! 👇\n\n#viral #trending #amazing #content #video",
        "twitter_post": f"🤯 {title[:200]} #viral #trending",
        "pinterest_description": f"Discover this amazing content: {title}. Perfect for inspiration!",
        "hashtags": {
            "youtube": ["#shorts", "#viral", "#trending", "#amazing"],
            "instagram": ["#reels", "#viral", "#trending", "#content"],
            "twitter": ["#viral", "#trending", "#content"],
            "pinterest": ["#viral", "#amazing", "#content"]
        }
    }

def extract_content(ai_text, content_type):
    """Extract specific content from AI response"""
    lines = ai_text.split('\n')
    for i, line in enumerate(lines):
        if content_type.lower() in line.lower():
            # Try to get the next line or current line content
            if i + 1 < len(lines):
                return lines[i + 1].strip()
            return line.split(':', 1)[-1].strip() if ':' in line else None
    return None

def cleanup_old_files():
    """Clean up old files to save disk space"""
    try:
        # Clean files older than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for folder in ['downloads', 'clips', 'temp']:
            if os.path.exists(folder):
                for file_path in Path(folder).glob('*'):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
    except Exception as e:
        print(f"Cleanup error: {e}")

# === BACKGROUND CLEANUP TASK ===
@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup"""
    cleanup_old_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))