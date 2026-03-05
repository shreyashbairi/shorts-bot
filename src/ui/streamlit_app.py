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
        disabled=not enable_captions,
        help="Viral: Large, centered, word highlight\nBold: Large uppercase\nMinimal: Clean bottom captions\nSubtitle: Traditional with background\nKaraoke: Green highlight effect"
    )
    caption_style_map = {
        "Viral": CaptionStyle.VIRAL,
        "Bold": CaptionStyle.BOLD,
        "Minimal": CaptionStyle.MINIMAL,
        "Subtitle": CaptionStyle.SUBTITLE,
        "Karaoke": CaptionStyle.KARAOKE,
    }

    # Caption customization expander
    with st.sidebar.expander("Caption Customization", expanded=False):
        caption_font = st.selectbox(
            "Font",
            options=["Arial-Bold", "Impact", "Helvetica-Bold", "Verdana-Bold", "Futura-Bold"],
            index=0,
            disabled=not enable_captions
        )

        caption_size = st.slider(
            "Font Size",
            min_value=32,
            max_value=96,
            value=60,
            step=4,
            disabled=not enable_captions
        )

        caption_position = st.selectbox(
            "Position",
            options=["center", "bottom", "top"],
            index=0,
            disabled=not enable_captions,
            help="Where captions appear on video"
        )

        caption_uppercase = st.checkbox(
            "Uppercase",
            value=True,
            disabled=not enable_captions
        )

        caption_highlight = st.checkbox(
            "Highlight Current Word",
            value=True,
            disabled=not enable_captions,
            help="Highlight the currently spoken word (viral effect)"
        )

        col1, col2 = st.columns(2)
        with col1:
            font_color = st.color_picker(
                "Font Color",
                value="#FFFFFF",
                disabled=not enable_captions
            )
        with col2:
            highlight_color = st.color_picker(
                "Highlight Color",
                value="#FFFF00",
                disabled=not enable_captions
            )

        stroke_width = st.slider(
            "Outline Width",
            min_value=0,
            max_value=8,
            value=3,
            disabled=not enable_captions
        )

        max_words = st.slider(
            "Max Words Per Line",
            min_value=2,
            max_value=8,
            value=4,
            disabled=not enable_captions
        )

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

        st.markdown("---")
        st.markdown("**LLM Highlight Analysis**")

        use_llm = st.checkbox(
            "Use LLM for Better Highlights",
            value=False,
            help="Use local Llama 3.1 model to improve highlight detection (requires model download)"
        )

        if use_llm:
            llm_backend = st.selectbox(
                "LLM Backend",
                options=["llama_cpp", "ollama"],
                index=0,
                help="llama_cpp: Direct with Metal acceleration (faster on M2)\nollama: Uses Ollama service"
            )

            llm_model_path = st.text_input(
                "Model Path (GGUF file)",
                value="models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                help="Path to the GGUF model file (Q5_K_M recommended for quality)"
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

    # Caption settings
    if enable_captions:
        config.caption.style = caption_style_map[caption_style]
        # Apply customizations (these override style preset defaults)
        config.caption.font_name = caption_font
        config.caption.font_size = caption_size
        config.caption.position = caption_position
        config.caption.uppercase = caption_uppercase
        config.caption.highlight_current_word = caption_highlight
        config.caption.font_color = font_color
        config.caption.highlight_color = highlight_color
        config.caption.stroke_width = stroke_width
        config.caption.max_words_per_line = max_words

    if language:
        config.transcription.language = language

    # LLM settings
    if use_llm:
        config.highlight.use_llm = True
        config.highlight.llm_backend = llm_backend
        if llm_model_path:
            config.highlight.llm_model_path = llm_model_path
        # Apple M2 optimizations
        config.highlight.llm_n_gpu_layers = -1  # All layers on Metal
        config.highlight.llm_n_threads = 6  # Good for M2
        config.highlight.llm_n_ctx = 8192  # Full context for transcript analysis
    else:
        config.highlight.use_llm = False

    st.session_state.config = config
    st.session_state.enable_captions = enable_captions

    # Show GPU/Metal status in sidebar
    gpu_available = get_default_gpu_enabled()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")

    # Check for Apple Silicon
    import platform as plat
    is_apple_silicon = plat.processor() == 'arm' and plat.system() == 'Darwin'

    if gpu_available:
        st.sidebar.success("GPU: CUDA Available")
    elif is_apple_silicon:
        st.sidebar.success("GPU: Apple Metal (M-series)")
    else:
        st.sidebar.info("GPU: Using CPU")

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

        # Show active configuration
        config = st.session_state.config
        if config:
            llm_status = "LLM: On" if config.highlight.use_llm else "LLM: Off"
            caption_status = f"Captions: {config.caption.style.value}" if st.session_state.get('enable_captions') else "Captions: Off"
            st.caption(f"Settings: {config.highlight.target_clips} clips | {config.highlight.min_clip_duration}-{config.highlight.max_clip_duration}s | Whisper: {config.transcription.model.value} | {caption_status} | {llm_status}")

        st.markdown("---")

        # Target duration option
        st.markdown("### Output Settings")
        col1, col2 = st.columns(2)
        with col1:
            target_duration_enabled = st.checkbox(
                "Specify target clip duration",
                value=False,
                help="Generate clips of approximately this length instead of auto-detecting"
            )
        with col2:
            target_duration = st.number_input(
                "Target Duration (seconds)",
                min_value=10,
                max_value=180,
                value=45,
                step=5,
                disabled=not target_duration_enabled,
                help="Each clip will be approximately this length"
            )

        if target_duration_enabled and config:
            config.highlight.min_clip_duration = max(10, target_duration - 10)
            config.highlight.max_clip_duration = target_duration + 10

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
    """Process the uploaded video with detailed progress tracking."""
    st.session_state.processing = True
    st.session_state.result = None

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        config = st.session_state.config

        # Create progress container with stage-based tracking
        progress_container = st.container()

        # Stage weights for overall progress calculation
        stage_weights = {
            PipelineStage.INPUT: 0.05,
            PipelineStage.TRANSCRIPTION: 0.30,
            PipelineStage.HIGHLIGHT_DETECTION: 0.15,
            PipelineStage.CLIPPING: 0.25,
            PipelineStage.CAPTIONING: 0.20,
            PipelineStage.EXPORT: 0.05,
        }

        stage_labels = {
            PipelineStage.INPUT: "Preparing input...",
            PipelineStage.TRANSCRIPTION: "Transcribing audio (this may take a while)...",
            PipelineStage.HIGHLIGHT_DETECTION: "Finding interesting moments...",
            PipelineStage.CLIPPING: "Creating video clips...",
            PipelineStage.CAPTIONING: "Adding captions...",
            PipelineStage.EXPORT: "Exporting final videos...",
            PipelineStage.COMPLETE: "Done!",
        }

        with progress_container:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            stage_text = st.empty()

        def update_progress(stage: PipelineStage, progress: float, message: str):
            # Calculate overall progress based on stage weights
            stages = list(stage_weights.keys())
            if stage in stages:
                stage_idx = stages.index(stage)
                completed_weight = sum(
                    stage_weights[s] for s in stages[:stage_idx]
                )
                current_weight = stage_weights[stage]
                overall = completed_weight + current_weight * progress
            elif stage == PipelineStage.COMPLETE:
                overall = 1.0
            else:
                overall = progress

            progress_bar.progress(min(overall, 1.0))
            status_text.text(message)
            stage_text.text(stage_labels.get(stage, f"Stage: {stage.value}"))

        # Create pipeline and process
        pipeline = Pipeline(config, progress_callback=update_progress)

        with st.spinner("Processing video..."):
            result = pipeline.process_file(tmp_path)

        st.session_state.result = result

        if result.success:
            st.success(f"Processing complete! Generated {len(result.clips)} clips in {result.processing_time:.1f}s")
        else:
            st.error(f"Processing failed: {result.error}")
            if result.error:
                with st.expander("Error Details"):
                    st.code(result.error)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.exception("Processing error")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())

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
            # Build expander title with virality score if available
            virality = clip.highlight.metadata.get('virality_score')
            title_suffix = f" | Virality: {virality}/100" if virality else ""
            suggested_title = clip.highlight.metadata.get('suggested_title', '')
            clip_label = suggested_title or f"Clip {i + 1}"

            with st.expander(f"{clip_label}: {clip.duration:.1f}s{title_suffix}", expanded=(i == 0)):
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

                    # Virality score with color coding
                    if virality is not None:
                        if virality >= 75:
                            st.success(f"Virality Score: {virality}/100")
                        elif virality >= 50:
                            st.info(f"Virality Score: {virality}/100")
                        else:
                            st.warning(f"Virality Score: {virality}/100")

                        explanation = clip.highlight.metadata.get('virality_explanation', '')
                        if explanation:
                            st.caption(explanation)

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

                # LLM-generated metadata (title, description, hashtags)
                if any(clip.highlight.metadata.get(k) for k in ['suggested_title', 'suggested_description', 'suggested_hashtags', 'hook_text']):
                    st.markdown("**AI-Generated Metadata**")
                    meta_cols = st.columns(2)
                    with meta_cols[0]:
                        if suggested_title:
                            st.text_input("Suggested Title", value=suggested_title, key=f"title_{i}", disabled=True)
                        desc = clip.highlight.metadata.get('suggested_description', '')
                        if desc:
                            st.text_area("Description", value=desc, key=f"desc_{i}", height=68, disabled=True)
                    with meta_cols[1]:
                        hashtags = clip.highlight.metadata.get('suggested_hashtags', [])
                        if hashtags:
                            st.write("**Hashtags:** " + " ".join(f"#{t}" for t in hashtags))
                        hook = clip.highlight.metadata.get('hook_text', '')
                        if hook:
                            st.write(f"**Hook Text:** {hook}")

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
    st.info("Uses configuration settings from sidebar (platform, caption style, LLM, etc.)")

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
                # Use shared config from sidebar
                config = st.session_state.config or Config()
                pipeline = Pipeline(config)

                llm_status = " (with LLM)" if config.highlight.use_llm else ""
                with st.spinner(f"Analyzing video{llm_status}..."):
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
                            reasons_text = ", ".join(r.value for r in h.reasons[:2]) if h.reasons else "general"
                            st.write(reasons_text)
                            if h.metadata.get('llm_score'):
                                st.caption(f"LLM: {h.metadata['llm_score']:.1f}")
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Preview error")

            finally:
                try:
                    tmp_path.unlink()
                except:
                    pass


def render_manual_clip_tab():
    """Render the manual clipping tab with support for multiple segments."""
    st.markdown("## Manual Clip")
    st.write("Create clips with custom start and end times. Multiple segments will be concatenated into one final video.")

    # Show active configuration
    config = st.session_state.config
    if config:
        caption_status = f"Captions: {config.caption.style.value}" if st.session_state.get('enable_captions') else "Captions: Off"
        preset_info = f"Quality: {config.clipping.preset} | Whisper: {config.transcription.model.value} | {caption_status}"
        st.info(f"Using sidebar settings: {preset_info}")

    uploaded_file = st.file_uploader(
        "Upload video",
        type=['mp4', 'mkv', 'avi', 'mov', 'webm'],
        key="manual_upload"
    )

    if uploaded_file:
        # LLM-powered cohesive clip creation
        config = st.session_state.config
        llm_available = config and config.highlight.use_llm

        if llm_available:
            st.markdown("### AI-Powered Clip Creation")
            st.write("Let the LLM analyze your video and create a cohesive clip by selecting the best segments.")

            col1, col2 = st.columns(2)
            with col1:
                target_duration = st.number_input(
                    "Target Duration (seconds)",
                    min_value=15,
                    max_value=180,
                    value=60,
                    help="The LLM will select segments that add up to approximately this duration"
                )
            with col2:
                ai_captions = st.checkbox("Add Captions", value=True, key="ai_captions")

            if st.button("🤖 Create AI-Powered Clip", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)

                try:
                    pipeline = Pipeline(config)

                    with st.spinner("Transcribing and analyzing video with LLM..."):
                        # First transcribe
                        input_file = pipeline.input_handler.process_file(tmp_path)
                        if input_file.audio_path:
                            transcript = pipeline.transcriber.transcribe(input_file.audio_path)

                            # Use LLM to create cohesive clip plan
                            segments = pipeline.highlight_detector.create_cohesive_clip(
                                transcript,
                                target_duration=float(target_duration)
                            )

                            if segments:
                                st.success(f"LLM selected {len(segments)} segments for a cohesive narrative")

                                # Show what the LLM selected
                                for i, seg in enumerate(segments):
                                    purpose = seg.get('purpose', '')
                                    st.write(f"**Segment {i+1}**: {seg['start']:.1f}s - {seg['end']:.1f}s ({purpose})")

                                # Create the clip
                                with st.spinner("Creating combined clip..."):
                                    clip = pipeline.create_multi_segment_clip(
                                        file_path=tmp_path,
                                        segments=segments,
                                        add_captions=ai_captions
                                    )

                                st.success(f"Clip created! Duration: {clip.duration:.1f}s")

                                if clip.path.exists():
                                    with open(clip.path, 'rb') as f:
                                        st.video(f.read())
                                    with open(clip.path, 'rb') as f:
                                        st.download_button("⬇️ Download AI Clip", f, file_name=clip.path.name, mime="video/mp4")
                            else:
                                st.error("LLM could not create a cohesive clip. Try manual selection below.")
                        else:
                            st.error("Could not extract audio from video")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("AI clip creation error")
                finally:
                    try:
                        tmp_path.unlink()
                    except:
                        pass

            st.markdown("---")
            st.markdown("### Or Create Manual Clip")
        else:
            st.info("Enable 'Use LLM for Better Highlights' in sidebar to unlock AI-powered clip creation")

        # Initialize segment state
        if 'num_segments' not in st.session_state:
            st.session_state.num_segments = 1
        if 'segments' not in st.session_state:
            st.session_state.segments = [{'start': 0.0, 'end': 30.0}]

        # Number of segments selector
        num_segments = st.number_input(
            "Number of Clips to Concatenate",
            min_value=1,
            max_value=20,
            value=st.session_state.num_segments,
            step=1,
            help="Specify how many segments you want to combine into one video"
        )

        # Update segments list if number changed
        if num_segments != st.session_state.num_segments:
            st.session_state.num_segments = num_segments
            # Adjust segments list
            while len(st.session_state.segments) < num_segments:
                last_end = st.session_state.segments[-1]['end'] if st.session_state.segments else 0
                st.session_state.segments.append({'start': last_end, 'end': last_end + 30.0})
            st.session_state.segments = st.session_state.segments[:num_segments]

        st.markdown("---")
        st.markdown("### Segment Times")

        # Display segment inputs
        segments = []
        total_duration = 0
        all_valid = True

        for i in range(num_segments):
            st.markdown(f"**Segment {i + 1}**")
            col1, col2, col3 = st.columns([2, 2, 1])

            # Get current values
            current_start = st.session_state.segments[i]['start'] if i < len(st.session_state.segments) else 0.0
            current_end = st.session_state.segments[i]['end'] if i < len(st.session_state.segments) else 30.0

            with col1:
                start_time = st.number_input(
                    f"Start (seconds)",
                    min_value=0.0,
                    value=current_start,
                    step=0.5,
                    key=f"start_{i}"
                )
            with col2:
                end_time = st.number_input(
                    f"End (seconds)",
                    min_value=0.0,
                    value=current_end,
                    step=0.5,
                    key=f"end_{i}"
                )
            with col3:
                duration = end_time - start_time
                if duration > 0:
                    st.metric("Duration", f"{duration:.1f}s")
                else:
                    st.error("Invalid")
                    all_valid = False

            # Update session state
            if i < len(st.session_state.segments):
                st.session_state.segments[i] = {'start': start_time, 'end': end_time}
            else:
                st.session_state.segments.append({'start': start_time, 'end': end_time})

            if end_time > start_time:
                segments.append({'start': start_time, 'end': end_time})
                total_duration += duration

        st.markdown("---")

        # Show total duration
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Total combined duration: **{total_duration:.1f} seconds**")
        with col2:
            # Default to sidebar setting
            default_captions = st.session_state.get('enable_captions', True)
            add_captions = st.checkbox("Add Captions", value=default_captions, key="manual_captions")

        # Create clip button
        if st.button("✂️ Create Combined Clip", type="primary", disabled=not all_valid):
            if not all_valid or not segments:
                st.error("Please fix invalid segments (end time must be greater than start time)")
                return

            # Validate segments
            for i, seg in enumerate(segments):
                if seg['end'] <= seg['start']:
                    st.error(f"Segment {i + 1}: End time must be greater than start time")
                    return

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                config = st.session_state.config or Config()
                pipeline = Pipeline(config)

                with st.spinner(f"Creating {len(segments)} segment(s) and concatenating..."):
                    if len(segments) == 1:
                        # Single segment - use existing method
                        clip = pipeline.create_single_clip(
                            file_path=tmp_path,
                            start_time=segments[0]['start'],
                            end_time=segments[0]['end'],
                            add_captions=add_captions
                        )
                    else:
                        # Multiple segments - use new method
                        clip = pipeline.create_multi_segment_clip(
                            file_path=tmp_path,
                            segments=segments,
                            add_captions=add_captions
                        )

                st.success(f"Clip created! Total duration: {clip.duration:.1f}s ({len(segments)} segment(s))")

                # Show and download
                if clip.path.exists():
                    with open(clip.path, 'rb') as f:
                        video_bytes = f.read()
                        st.video(video_bytes)

                    with open(clip.path, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Combined Clip",
                            f,
                            file_name=clip.path.name,
                            mime="video/mp4"
                        )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Manual clip error")

            finally:
                try:
                    tmp_path.unlink()
                except:
                    pass

        # Help section
        with st.expander("How to use multiple segments"):
            st.markdown("""
            1. **Set the number of clips** you want to combine
            2. **Enter start and end times** for each segment
            3. Segments will be **concatenated in order** (Segment 1, then Segment 2, etc.)
            4. Click **Create Combined Clip** to generate the final video

            **Tips:**
            - Segments don't need to be in chronological order
            - You can pick any parts of the video to combine
            - The final video will play segments in the order listed (1, 2, 3...)
            - Use the Quick Preview tab to find interesting timestamps first
            """)


def render_multi_video_tab():
    """Render the multi-video processing tab."""
    st.markdown("## Multi-Video Processing")
    st.write("Upload multiple videos, find the best highlights across all of them, and stitch them into one clip.")

    uploaded_files = st.file_uploader(
        "Upload multiple videos",
        type=['mp4', 'mkv', 'avi', 'mov', 'webm'],
        accept_multiple_files=True,
        key="multi_video_upload"
    )

    if uploaded_files and len(uploaded_files) >= 2:
        st.info(f"{len(uploaded_files)} videos selected")

        for i, f in enumerate(uploaded_files):
            size_mb = f.size / (1024 * 1024)
            st.write(f"**{i+1}.** {f.name} ({size_mb:.1f} MB)")

        st.markdown("---")
        st.markdown("### Processing Options")

        col1, col2 = st.columns(2)
        with col1:
            stitch_mode = st.selectbox(
                "Highlight Selection Mode",
                options=["Best Overall", "One Per Video"],
                index=0,
                help="Best Overall: Pick the best highlights across all videos\n"
                     "One Per Video: Pick the best highlight from each video"
            )
            mode_map = {"Best Overall": "best_highlights", "One Per Video": "one_per_video"}

        with col2:
            target_dur_enabled = st.checkbox("Set target duration", value=False)
            target_dur = st.number_input(
                "Target Duration (s)",
                min_value=15,
                max_value=300,
                value=60,
                disabled=not target_dur_enabled
            )

        col3, col4 = st.columns(2)
        with col3:
            mv_transition = st.selectbox(
                "Transition Between Clips",
                options=["None", "Crossfade", "Fade to Black", "Wipe"],
                index=1,
                key="mv_transition"
            )
            mv_trans_map = {"None": None, "Crossfade": "crossfade", "Fade to Black": "fade_black", "Wipe": "wipe"}

        with col4:
            mv_trans_dur = st.slider(
                "Transition Duration (s)",
                min_value=0.2,
                max_value=2.0,
                value=0.5,
                step=0.1,
                disabled=(mv_transition == "None"),
                key="mv_trans_dur"
            )

        mv_captions = st.checkbox("Add Captions", value=True, key="mv_captions")

        st.markdown("---")

        if st.button("🎬 Process & Stitch Videos", type="primary"):
            # Save all uploaded files to temp
            temp_paths = []
            try:
                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                        tmp.write(uf.getvalue())
                        temp_paths.append(Path(tmp.name))

                config = st.session_state.config or Config()

                # Progress tracking
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(stage, progress, message):
                    stage_weights = {
                        PipelineStage.INPUT: 0.05,
                        PipelineStage.TRANSCRIPTION: 0.35,
                        PipelineStage.HIGHLIGHT_DETECTION: 0.15,
                        PipelineStage.CLIPPING: 0.20,
                        PipelineStage.CAPTIONING: 0.15,
                        PipelineStage.EXPORT: 0.10,
                    }
                    stages = list(stage_weights.keys())
                    if stage in stages:
                        idx = stages.index(stage)
                        completed = sum(stage_weights[s] for s in stages[:idx])
                        overall = completed + stage_weights[stage] * progress
                    elif stage == PipelineStage.COMPLETE:
                        overall = 1.0
                    else:
                        overall = progress
                    progress_bar.progress(min(overall, 1.0))
                    status_text.text(message)

                pipeline = Pipeline(config, progress_callback=update_progress)

                with st.spinner(f"Processing {len(temp_paths)} videos..."):
                    result = pipeline.process_multi_video(
                        file_paths=temp_paths,
                        stitch_mode=mode_map[stitch_mode],
                        target_duration=float(target_dur) if target_dur_enabled else None,
                        transition=mv_trans_map[mv_transition],
                        transition_duration=mv_trans_dur,
                        add_captions=mv_captions,
                    )

                if result.success and result.clips:
                    clip = result.clips[0]
                    st.success(
                        f"Multi-video processing complete! "
                        f"Duration: {clip.duration:.1f}s | "
                        f"Processing time: {result.processing_time:.1f}s"
                    )

                    if clip.path.exists():
                        with open(clip.path, 'rb') as f:
                            st.video(f.read())
                        with open(clip.path, 'rb') as f:
                            st.download_button(
                                "⬇️ Download Stitched Video",
                                f,
                                file_name="multi_video_result.mp4",
                                mime="video/mp4"
                            )
                else:
                    st.error(f"Processing failed: {result.error}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Multi-video processing error")
            finally:
                for p in temp_paths:
                    try:
                        p.unlink()
                    except Exception:
                        pass

    elif uploaded_files and len(uploaded_files) < 2:
        st.warning("Please upload at least 2 videos for multi-video processing. For a single video, use the 'Process Video' tab.")
    else:
        st.info("Upload 2 or more videos to get started. The app will find the best moments across all videos and combine them into one clip.")


def render_stitch_clips_tab():
    """Render the stitch clips tab for combining multiple generated clips."""
    st.markdown("## Stitch Clips")
    st.write("Combine multiple clips into one longer video. Upload clips or use previously generated ones.")

    # Transition settings
    st.markdown("### Transition Settings")
    trans_col1, trans_col2 = st.columns(2)
    with trans_col1:
        transition_type = st.selectbox(
            "Transition Effect",
            options=["None", "Crossfade", "Fade to Black", "Wipe"],
            index=0,
            help="Effect applied between clips when stitching"
        )
    with trans_col2:
        transition_duration = st.slider(
            "Transition Duration (s)",
            min_value=0.2,
            max_value=2.0,
            value=0.5,
            step=0.1,
            disabled=(transition_type == "None")
        )

    transition_map = {
        "None": None,
        "Crossfade": "crossfade",
        "Fade to Black": "fade_black",
        "Wipe": "wipe",
    }
    selected_transition = transition_map[transition_type]

    st.markdown("---")

    # Method selection
    stitch_method = st.radio(
        "Select clips to stitch",
        options=["Upload clip files", "Use generated clips from last processing"],
        horizontal=True
    )

    clip_paths = []

    if stitch_method == "Upload clip files":
        uploaded_clips = st.file_uploader(
            "Upload video clips to combine",
            type=['mp4', 'mkv', 'avi', 'mov', 'webm'],
            accept_multiple_files=True,
            key="stitch_upload"
        )

        if uploaded_clips:
            st.info(f"{len(uploaded_clips)} clips selected")

            # Show clip order
            st.markdown("### Clip Order")
            st.write("Clips will be combined in the order shown below. Drag to reorder (in future).")

            for i, clip in enumerate(uploaded_clips):
                st.write(f"**{i + 1}.** {clip.name} ({clip.size / 1024:.0f} KB)")

            if st.button("🔗 Stitch Clips Together", type="primary"):
                # Save uploaded clips to temp files
                temp_paths = []
                try:
                    for clip in uploaded_clips:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(clip.getvalue())
                            temp_paths.append(Path(tmp.name))

                    config = st.session_state.config or Config()
                    from src.modules.video_clipper import VideoClipper
                    clipper = VideoClipper(config)

                    with st.spinner(f"Stitching {len(temp_paths)} clips together..."):
                        output_path = clipper.concatenate_clips(
                            clip_paths=temp_paths,
                            output_prefix="stitched_clip",
                            transition=selected_transition,
                            transition_duration=transition_duration
                        )

                    if output_path.exists():
                        st.success(f"Clips stitched successfully!")
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                "⬇️ Download Stitched Video",
                                f,
                                file_name="stitched_video.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("Failed to create stitched video")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Stitch error")
                finally:
                    for p in temp_paths:
                        try:
                            p.unlink()
                        except:
                            pass

    else:
        # Use generated clips from last processing
        result = st.session_state.result
        if result and result.clips:
            st.success(f"{len(result.clips)} clips available from last processing")

            # Let user select which clips to include and in what order
            st.markdown("### Select and Order Clips")
            selected_clips = []

            for i, clip in enumerate(result.clips):
                col1, col2, col3 = st.columns([0.5, 2, 1])
                with col1:
                    include = st.checkbox(f"Include", value=True, key=f"stitch_include_{i}")
                with col2:
                    st.write(f"**Clip {i + 1}**: {clip.duration:.1f}s (Score: {clip.highlight.score:.2f})")
                with col3:
                    if clip.path.exists():
                        st.write(f"{clip.file_size // 1024} KB")

                if include and clip.path.exists():
                    selected_clips.append(clip)

            if selected_clips:
                total_dur = sum(c.duration for c in selected_clips)
                st.info(f"Selected {len(selected_clips)} clips, total duration: {total_dur:.1f}s")

                if st.button("🔗 Stitch Selected Clips", type="primary"):
                    try:
                        config = st.session_state.config or Config()
                        from src.modules.video_clipper import VideoClipper
                        clipper = VideoClipper(config)

                        with st.spinner(f"Stitching {len(selected_clips)} clips..."):
                            output_path = clipper.concatenate_clips(
                                clip_paths=[c.path for c in selected_clips],
                                output_prefix="stitched_result",
                                transition=selected_transition,
                                transition_duration=transition_duration
                            )

                        if output_path.exists():
                            st.success("Clips stitched successfully!")
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    "⬇️ Download Stitched Video",
                                    f,
                                    file_name="stitched_result.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.error("Failed to create stitched video")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.exception("Stitch error")
            else:
                st.warning("No clips selected")
        else:
            st.info("No clips available. Process a video first in the 'Process Video' tab, then return here to stitch clips together.")


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    config = render_sidebar()

    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎬 Process Video", "🔍 Quick Preview", "✂️ Manual Clip",
        "📹 Multi-Video", "🔗 Stitch Clips", "ℹ️ About"
    ])

    with tab1:
        render_main_content()

    with tab2:
        render_preview_tab()

    with tab3:
        render_manual_clip_tab()

    with tab4:
        render_multi_video_tab()

    with tab5:
        render_stitch_clips_tab()

    with tab6:
        st.markdown("""
        ## About Shorts Bot

        Shorts Bot is a free, open-source tool for converting long-form videos
        into short-form vertical content optimized for:

        - **YouTube Shorts** (up to 60 seconds)
        - **Instagram Reels** (up to 90 seconds)
        - **TikTok** (up to 3 minutes)

        ### Features

        - **Automatic Highlight Detection** - AI identifies the most engaging moments
        - **LLM-Powered Scoring** - Context-aware segment analysis (replaces keyword matching)
        - **Virality Score** - Each clip gets a 0-100 viral potential prediction
        - **AI Metadata** - Auto-generated titles, descriptions, hashtags, and hook text
        - **Multi-Video Processing** - Upload multiple videos, stitch best highlights
        - **Transition Effects** - Crossfade, fade-to-black, and wipe between clips
        - **Speech-to-Text** - Accurate transcription using OpenAI Whisper
        - **Auto-Captions** - Viral-style animated captions with LLM emoji placement
        - **Smart Framing** - Face detection for optimal cropping
        - **Multiple Styles** - Choose from various caption presets

        ### How It Works

        1. Upload your long-form video (or multiple videos)
        2. Configure your preferences
        3. Click "Process Video"
        4. Review clips with virality scores and AI-suggested metadata
        5. Download your ready-to-post shorts!

        ### Technical Details

        - **Transcription**: OpenAI Whisper (local, no API needed)
        - **Video Processing**: FFmpeg with xfade transitions
        - **Face Detection**: OpenCV (DNN + Haar cascade)
        - **LLM Intelligence**: Local Llama 3.1 via llama-cpp (Metal on Apple Silicon)
        - **100% Local** - Your videos never leave your computer

        ---

        Made with Python, Streamlit, and open-source tools.
        """)

        # Show current config
        with st.expander("Current Configuration"):
            if st.session_state.config:
                st.json(st.session_state.config.to_dict())


if __name__ == "__main__":
    main()
