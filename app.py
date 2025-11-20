import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎬",
    layout="wide"
)

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_title(video_id):
    """Get video title from YouTube"""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        return title.text.replace(' - YouTube', '') if title else "Unknown Title"
    except:
        return "Unknown Title"

def get_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = ' '.join([item['text'] for item in transcript])
        return full_text, len(transcript)
    except Exception as e:
        return None, 0

def generate_summary(text, max_length=150, min_length=30):
    """Generate summary using Hugging Face pipeline"""
    try:
        # Use a smaller, faster model for summarization
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        
        # Split text if it's too long (BART has token limits)
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summary generation failed: {str(e)}. Here are the first 500 characters: {text[:500]}..."

def main():
    st.title("🎬 YouTube Video Summarizer")
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/..."
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        summarize_btn = st.button("🚀 Summarize Video", use_container_width=True)
    
    st.markdown("---")
    
    if summarize_btn:
        if not youtube_url:
            st.warning("⚠️ Please enter a YouTube URL")
            return
        
        with st.spinner("🔄 Processing video..."):
            # Extract video ID
            video_id = extract_video_id(youtube_url)
            
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please check the link and try again.")
                return
            
            # Get video title
            video_title = get_video_title(video_id)
            
            # Display video info
            st.subheader("📹 Video Information")
            info_col1, info_col2 = st.columns([2, 1])
            
            with info_col1:
                st.write(f"**Title:** {video_title}")
                st.write(f"**Video ID:** {video_id}")
                st.video(youtube_url)
            
            # Get transcript
            transcript, sentence_count = get_transcript(video_id)
            
            if not transcript:
                st.error("❌ Could not retrieve transcript. This video might not have captions available.")
                return
            
            with info_col2:
                st.metric("Sentences in Transcript", sentence_count)
                st.metric("Characters in Transcript", len(transcript))
            
            st.markdown("---")
            
            # Generate and display summary
            st.subheader("📝 Summary")
            
            with st.spinner("🤖 Generating summary..."):
                summary = generate_summary(transcript)
            
            st.success("✅ Summary generated successfully!")
            st.text_area("Summary", summary, height=150, key="summary")
            
            st.markdown("---")
            
            # Display full transcript
            st.subheader("📄 Full Transcript")
            with st.expander("View Full Transcript", expanded=False):
                st.text_area("Transcript", transcript, height=300, key="transcript")
            
            # Download buttons
            st.markdown("---")
            st.subheader("💾 Download")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"summary_{video_id}.txt",
                    mime="text/plain"
                )
            
            with download_col2:
                st.download_button(
                    label="📥 Download Full Transcript",
                    data=transcript,
                    file_name=f"transcript_{video_id}.txt",
                    mime="text/plain"
                )

    else:
        # Display instructions when no URL is entered
        st.info("👆 Enter a YouTube URL above and click 'Summarize Video' to get started!")
        
        st.markdown("""
        ### 📋 How to use:
        1. **Copy** a YouTube video URL
        2. **Paste** it in the input field above
        3. **Click** the 'Summarize Video' button
        4. **Wait** for the AI to process and summarize the content
        
        ### ⚠️ Requirements:
        - Video must have captions/subtitles available
        - Supported URL formats:
          - `https://www.youtube.com/watch?v=VIDEO_ID`
          - `https://youtu.be/VIDEO_ID`
          - YouTube Shorts URLs
        """)

if __name__ == "__main__":
    main()