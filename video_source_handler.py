import cv2
import requests
import re
import logging
from urllib.parse import urlparse, parse_qs
import json

logger = logging.getLogger(__name__)

class VideoSourceHandler:
    """Enhanced video source handler for various platforms and formats"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_direct_video_url(self, url):
        """Check if URL is a direct video file"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        return any(url.lower().endswith(ext) for ext in video_extensions)
    
    def is_streaming_url(self, url):
        """Check if URL is a streaming protocol"""
        streaming_protocols = ['rtmp://', 'rtsp://', 'http://127.0.0.1', 'http://localhost']
        return any(url.startswith(protocol) for protocol in streaming_protocols)
    
    def extract_pexels_video_url(self, page_url):
        """Extract direct video URL from Pexels page"""
        try:
            response = self.session.get(page_url)
            response.raise_for_status()
            
            # Look for video URLs in the page content
            # Pexels typically has video URLs in script tags or data attributes
            content = response.text
            
            # Pattern to find video URLs
            patterns = [
                r'"url":"([^"]*\.mp4[^"]*)"',
                r'data-src="([^"]*\.mp4[^"]*)"',
                r'src="([^"]*\.mp4[^"]*)"',
                r'"link":"([^"]*\.mp4[^"]*)"'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Return the highest quality video URL (usually the longest URL)
                    video_url = max(matches, key=len)
                    # Clean up escaped characters
                    video_url = video_url.replace('\\/', '/')
                    logger.info(f"Extracted Pexels video URL: {video_url}")
                    return video_url
                    
        except Exception as e:
            logger.error(f"Failed to extract Pexels video URL: {e}")
        
        return None
    
    def extract_youtube_info(self, url):
        """Extract YouTube video info (requires yt-dlp)"""
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit to 720p for performance
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info['url']
                
        except ImportError:
            logger.error("yt-dlp not installed. Install with: pip install yt-dlp")
        except Exception as e:
            logger.error(f"Failed to extract YouTube URL: {e}")
        
        return None
    
    def get_video_source(self, source):
        """Get the appropriate video source for OpenCV"""
        # If it's a number (camera index)
        if isinstance(source, str) and source.isdigit():
            return int(source)
        
        # If it's already a number
        if isinstance(source, int):
            return source
        
        # If it's a local file path
        if not source.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
            return source
        
        # If it's a direct video URL or streaming URL
        if self.is_direct_video_url(source) or self.is_streaming_url(source):
            return source
        
        # Handle different platforms
        if 'pexels.com' in source:
            extracted_url = self.extract_pexels_video_url(source)
            if extracted_url:
                return extracted_url
        
        elif 'youtube.com' in source or 'youtu.be' in source:
            extracted_url = self.extract_youtube_info(source)
            if extracted_url:
                return extracted_url
        
        # If we can't extract, try to find video URLs in the page
        logger.warning(f"Attempting to extract video URL from: {source}")
        return self.generic_video_extraction(source)
    
    def generic_video_extraction(self, page_url):
        """Generic video URL extraction from web pages"""
        try:
            response = self.session.get(page_url)
            response.raise_for_status()
            content = response.text
            
            # Look for common video URL patterns
            video_patterns = [
                r'https?://[^"\s]+\.mp4[^"\s]*',
                r'https?://[^"\s]+\.webm[^"\s]*',
                r'https?://[^"\s]+\.mov[^"\s]*',
            ]
            
            for pattern in video_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Clean and return the first valid URL
                    for match in matches:
                        cleaned_url = match.replace('\\/', '/').strip('"\'')
                        if self.test_video_url(cleaned_url):
                            logger.info(f"Found video URL: {cleaned_url}")
                            return cleaned_url
                            
        except Exception as e:
            logger.error(f"Generic extraction failed: {e}")
        
        return None
    
    def test_video_url(self, url):
        """Test if a URL is a valid video source"""
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
        except Exception:
            pass
        return False
    
    def get_sample_urls(self):
        """Return some sample video URLs for testing"""
        return {
            'sample_mp4': 'https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4',
            'big_buck_bunny': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            'test_video': 'https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4'
        }

def get_video_source(source_input):
    """Convenience function to get video source"""
    handler = VideoSourceHandler()
    return handler.get_video_source(source_input)