"""
Streamlit Web Interface for Shorts Bot

A user-friendly web application for converting long-form videos
into short-form vertical content.

Run with: streamlit run src/ui/streamlit_app.py
"""

import streamlit as st
import tempfile
import time
from pathlib import Path
from typing import Optional
import json

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import (
    Config, Platform, CaptionStyle, WhisperModel,
    PlatformPreset, PRESET_CONFIGS, get_preset_config,
    get_default_gpu_enabled
)
from src.core.pipeline import Pipeline, PipelineStage
from src.core.logger import get_logger

logger = get_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Shorts Bot - Video Converter",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
    }
    .clip-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'processing': False,
        'current_stage': None,
        'progress': 0.0,
        'status_message': '',
        'result': None,
        'highlights': None,
        'config': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.markdown("## Configuration")

    # Platform selection
    platform = st.sidebar.selectbox(
        "Target Platform",
        options=["YouTube Shorts", "Instagram Reels", "TikTok"],
        index=0
    )
    platform_map = {
        "YouTube Shorts": Platform.YOUTUBE_SHORTS,
        "Instagram Reels": Platform.INSTAGRAM_REELS,
        "TikTok": Platform.TIKTOK,
    }

    st.sidebar.markdown("---")

    # Quality preset
    preset = st.sidebar.selectbox(
        "Quality Preset",
        options=["Fast", "Balanced", "Quality", "Max Quality"],
        index=1,
        help="Fast: Quick processing, lower quality\nBalanced: Good balance\nQuality: Higher quality, slower\nMax Quality: Best quality, slowest"
    )
    preset_map = {
        "Fast": "fast",
        "Balanced": "balanced",
        "Quality": "quality",
        "Max Quality": "max_quality",
    }

    st.sidebar.markdown("---")

    # Clip settings
    st.sidebar.markdown("### Clip Settings")

    num_clips = st.sidebar.slider(
        "Number of Clips",
        min_value=1,
        max_value=20,
        value=5,
        help="How many clips to generate from the video"
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_duration = st.number_input(
            "Min Duration (s)",
            min_value=5,
            max_value=60,
            value=15
        )
    with col2:
        max_duration = st.number_input(
            "Max Duration (s)",
            min_value=15,
            max_value=180,
            value=60
        )

    st.sidebar.markdown("---")

    # Caption settings
    st.sidebar.markdown("### Caption Settings")

    enable_captions = st.sidebar.checkbox("Enable Captions", value=True)

    caption_style = st.sidebar.selectbox(
        "Caption Style",
        options=["Viral", "Bold", "Minimal", "Subtitle", "Karaoke"],
        index=0,
        disabled=not enable_captions
    )
    caption_style_map = {
        "Viral": CaptionStyle.VIRAL,
        "Bold": CaptionStyle.BOLD,
        "Minimal": CaptionStyle.MINIMAL,
        "Subtitle": CaptionStyle.SUBTITLE,
        "Karaoke": CaptionStyle.KARAOKE,
    }

    st.sidebar.markdown("---")

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        whisper_model = st.selectbox(
            "Whisper Model",
            options=["tiny", "base", "small", "medium", "large", "large-v3"],
            index=1,
            help="Larger models are more accurate but slower"
        )

        smart_framing = st.checkbox(
            "Smart Framing (Face Detection)",
            value=True,
            help="Automatically center the frame on detected faces"
        )

        language = st.text_input(
            "Language (optional)",
            placeholder="e.g., en, es, fr",
            help="Leave empty for auto-detection"
        )

    # Build config from settings (use get_preset_config for proper GPU detection)
    config = get_preset_config(preset_map[preset])
    config.update_for_platform(platform_map[platform])
    config.highlight.target_clips = num_clips
    config.highlight.min_clip_duration = min_duration
    config.highlight.max_clip_duration = max_duration
    config.transcription.model = WhisperModel(whisper_model)
    config.clipping.smart_framing = smart_framing
    config.clipping.face_detection = smart_framing

    if enable_captions:
        config.caption.style = caption_style_map[caption_style]

    if language:
        config.transcription.language = language

    st.session_state.config = config
    st.session_state.enable_captions = enable_captions

    # Show GPU status in sidebar
    gpu_available = get_default_gpu_enabled()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    if gpu_available:
        st.sidebar.success("GPU: CUDA Available")
    else:
        st.sidebar.info("GPU: Using CPU (no CUDA)")

    return config


def render_main_content():
    """Render the main content area."""
    st.markdown('<p class="main-header">🎬 Shorts Bot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Convert long-form videos into viral short-form content</p>',
        unsafe_allow_html=True
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your video",
        type=['mp4', 'mkv', 'avi', 'mov', 'webm', 'mp3', 'wav'],
        help="Supported formats: MP4, MKV, AVI, MOV, WebM, MP3, WAV"
    )

    if uploaded_file:
        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name[:30] + "..." if len(uploaded_file.name) > 30 else uploaded_file.name)
        with col2:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("File Size", f"{size_mb:.1f} MB")
        with col3:
            st.metric("Format", uploaded_file.type.split('/')[-1].upper())

        st.markdown("---")

        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🚀 Process Video",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing
            )

        if process_button:
            process_video(uploaded_file)

    # Show results if available
    if st.session_state.result:
        render_results()


def process_video(uploaded_file):
    """Process the uploaded video."""
    st.session_state.processing = True
    st.session_state.result = None

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        config = st.session_state.config

        # Create progress container
        progress_container = st.container()

        with progress_container:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            stage_text = st.empty()

        def update_progress(stage: PipelineStage, progress: float, message: str):
            progress_bar.progress(progress)
            status_text.text(message)
            stage_text.text(f"Stage: {stage.value}")

        # Create pipeline and process
        pipeline = Pipeline(config, progress_callback=update_progress)

        with st.spinner("Processing video..."):
            result = pipeline.process_file(tmp_path)

        st.session_state.result = result

        if result.success:
            st.success(f"Processing complete! Generated {len(result.clips)} clips in {result.processing_time:.1f}s")
        else:
            st.error(f"Processing failed: {result.error}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.exception("Processing error")

    finally:
        st.session_state.processing = False
        # Cleanup temp file
        try:
            tmp_path.unlink()
        except:
            pass


def render_results():
    """Render processing results."""
    result = st.session_state.result

    if not result or not result.success:
        return

    st.markdown("---")
    st.markdown("## Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clips Generated", len(result.clips))
    with col2:
        st.metric("Total Words", result.transcript.word_count if result.transcript else 0)
    with col3:
        st.metric("Source Duration", f"{result.input_file.duration:.0f}s")
    with col4:
        st.metric("Processing Time", f"{result.processing_time:.1f}s")

    st.markdown("---")

    # Clips
    if result.clips:
        st.markdown("### Generated Clips")

        for i, clip in enumerate(result.clips):
            with st.expander(f"Clip {i + 1}: {clip.duration:.1f}s", expanded=(i == 0)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Video preview (if file exists)
                    if clip.path.exists():
                        try:
                            with open(clip.path, 'rb') as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                        except Exception as e:
                            st.warning(f"Could not load video preview: {e}")

                with col2:
                    st.markdown("**Details**")
                    st.write(f"Duration: {clip.duration:.1f}s")
                    st.write(f"Size: {clip.file_size // 1024} KB")
                    st.write(f"Score: {clip.highlight.score:.2f}")

                    st.markdown("**Reasons**")
                    for reason in clip.highlight.reasons:
                        st.write(f"• {reason.value.replace('_', ' ').title()}")

                    # Download button
                    if clip.path.exists():
                        with open(clip.path, 'rb') as f:
                            st.download_button(
                                "⬇️ Download",
                                f,
                                file_name=clip.path.name,
                                mime="video/mp4"
                            )

    # Transcript preview
    if result.transcript:
        with st.expander("📝 Full Transcript"):
            st.text_area(
                "Transcript",
                value=result.transcript.text,
                height=200,
                disabled=True
            )

            # Download transcript
            transcript_json = json.dumps(result.transcript.to_dict(), indent=2)
            st.download_button(
                "⬇️ Download Transcript (JSON)",
                transcript_json,
                file_name="transcript.json",
                mime="application/json"
            )

    # Highlights
    if result.highlights:
        with st.expander("🎯 All Detected Highlights"):
            for i, h in enumerate(result.highlights.highlights):
                st.markdown(f"**{i + 1}. [{h.start:.1f}s - {h.end:.1f}s]** (Score: {h.score:.2f})")
                st.write(f"Reasons: {', '.join(r.value for r in h.reasons)}")
                st.write(f"Preview: \"{h.text[:150]}...\"" if len(h.text) > 150 else f"Text: \"{h.text}\"")
                st.markdown("---")


def render_preview_tab():
    """Render the highlight preview tab."""
    st.markdown("## Quick Preview")
    st.write("Get a quick preview of potential highlights without full processing.")

    uploaded_file = st.file_uploader(
        "Upload video for preview",
        type=['mp4', 'mkv', 'avi', 'mov', 'webm'],
        key="preview_upload"
    )

    if uploaded_file:
        if st.button("🔍 Analyze Highlights", type="secondary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                config = Config()
                pipeline = Pipeline(config)

                with st.spinner("Analyzing video (this may take a moment)..."):
                    highlights = pipeline.preview_highlights(tmp_path, quick_mode=True)

                st.success(f"Found {len(highlights)} potential highlights!")

                for i, h in enumerate(highlights, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Highlight {i}**")
                            st.write(f"Time: {h.start:.1f}s - {h.end:.1f}s ({h.duration:.1f}s)")
                            st.write(f"\"{h.text[:200]}...\"" if len(h.text) > 200 else f"\"{h.text}\"")
                        with col2:
                            st.metric("Score", f"{h.score:.2f}")
                            st.write(", ".join(r.value for r in h.reasons[:2]))
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error: {str(e)}")

            finally:
                try:
                    tmp_path.unlink()
                except:
                    pass


def render_manual_clip_tab():
    """Render the manual clipping tab."""
    st.markdown("## Manual Clip")
    st.write("Create a clip with custom start and end times.")

    uploaded_file = st.file_uploader(
        "Upload video",
        type=['mp4', 'mkv', 'avi', 'mov', 'webm'],
        key="manual_upload"
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start Time (seconds)", min_value=0.0, value=0.0, step=0.5)
        with col2:
            end_time = st.number_input("End Time (seconds)", min_value=0.0, value=30.0, step=0.5)

        add_captions = st.checkbox("Add Captions", value=True)

        if st.button("✂️ Create Clip", type="primary"):
            if end_time <= start_time:
                st.error("End time must be greater than start time")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                config = st.session_state.config or Config()
                pipeline = Pipeline(config)

                with st.spinner("Creating clip..."):
                    clip = pipeline.create_single_clip(
                        file_path=tmp_path,
                        start_time=start_time,
                        end_time=end_time,
                        add_captions=add_captions
                    )

                st.success(f"Clip created! Duration: {clip.duration:.1f}s")

                # Show and download
                if clip.path.exists():
                    with open(clip.path, 'rb') as f:
                        video_bytes = f.read()
                        st.video(video_bytes)

                        st.download_button(
                            "⬇️ Download Clip",
                            f,
                            file_name=clip.path.name,
                            mime="video/mp4"
                        )

            except Exception as e:
                st.error(f"Error: {str(e)}")

            finally:
                try:
                    tmp_path.unlink()
                except:
                    pass


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    config = render_sidebar()

    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Process Video", "🔍 Quick Preview", "✂️ Manual Clip", "ℹ️ About"])

    with tab1:
        render_main_content()

    with tab2:
        render_preview_tab()

    with tab3:
        render_manual_clip_tab()

    with tab4:
        st.markdown("""
        ## About Shorts Bot

        Shorts Bot is a free, open-source tool for converting long-form videos
        into short-form vertical content optimized for:

        - **YouTube Shorts** (up to 60 seconds)
        - **Instagram Reels** (up to 90 seconds)
        - **TikTok** (up to 3 minutes)

        ### Features

        - 🎯 **Automatic Highlight Detection** - AI identifies the most engaging moments
        - 🗣️ **Speech-to-Text** - Accurate transcription using OpenAI Whisper
        - 📝 **Auto-Captions** - Viral-style animated captions
        - 📷 **Smart Framing** - Face detection for optimal cropping
        - 🎨 **Multiple Styles** - Choose from various caption presets

        ### How It Works

        1. Upload your long-form video
        2. Configure your preferences
        3. Click "Process Video"
        4. Download your ready-to-post shorts!

        ### Technical Details

        - **Transcription**: OpenAI Whisper (local, no API needed)
        - **Video Processing**: FFmpeg
        - **Face Detection**: OpenCV
        - **100% Local** - Your videos never leave your computer

        ---

        Made with ❤️ using Python, Streamlit, and open-source tools.
        """)

        # Show current config
        with st.expander("Current Configuration"):
            if st.session_state.config:
                st.json(st.session_state.config.to_dict())


if __name__ == "__main__":
    main()
