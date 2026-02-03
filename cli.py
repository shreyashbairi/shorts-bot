#!/usr/bin/env python3
"""
Shorts Bot - Command Line Interface

Convert long-form videos into short-form vertical content for
YouTube Shorts, Instagram Reels, and TikTok.

Usage:
    python cli.py process video.mp4
    python cli.py process --batch ./videos/
    python cli.py preview video.mp4
    python cli.py clip video.mp4 --start 30 --end 60
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.core.config import (
    Config, Platform, CaptionStyle, WhisperModel,
    PRESET_CONFIGS, get_preset_config
)
from src.core.pipeline import Pipeline, PipelineStage
from src.core.logger import setup_global_logging, get_logger


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='shorts-bot',
        description='Convert long-form videos into short-form vertical content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process video.mp4                    Process a single video
  %(prog)s process --batch ./videos/            Process all videos in directory
  %(prog)s process video.mp4 --platform tiktok  Process for TikTok
  %(prog)s process video.mp4 --clips 10         Generate 10 clips
  %(prog)s preview video.mp4                    Quick preview of highlights
  %(prog)s clip video.mp4 --start 30 --end 60   Create manual clip
  %(prog)s transcribe video.mp4                 Transcribe only
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process video(s) into shorts')
    process_parser.add_argument('input', type=str, help='Input video file or directory')
    process_parser.add_argument('--batch', action='store_true', help='Batch process directory')
    process_parser.add_argument('--recursive', '-r', action='store_true', help='Recursive directory scan')
    process_parser.add_argument('--output', '-o', type=str, help='Output directory')
    process_parser.add_argument('--platform', '-p', type=str,
                               choices=['youtube_shorts', 'instagram_reels', 'tiktok'],
                               default='youtube_shorts', help='Target platform')
    process_parser.add_argument('--clips', '-n', type=int, default=5, help='Number of clips to generate')
    process_parser.add_argument('--min-duration', type=int, default=15, help='Minimum clip duration (seconds)')
    process_parser.add_argument('--max-duration', type=int, default=60, help='Maximum clip duration (seconds)')
    process_parser.add_argument('--preset', type=str, choices=['fast', 'balanced', 'quality', 'max_quality'],
                               help='Quality preset')
    process_parser.add_argument('--model', '-m', type=str,
                               choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                               default='base', help='Whisper model size')
    process_parser.add_argument('--language', '-l', type=str, help='Language code (e.g., en, es, fr)')
    process_parser.add_argument('--caption-style', type=str,
                               choices=['minimal', 'bold', 'viral', 'subtitle', 'karaoke'],
                               default='viral', help='Caption style')
    process_parser.add_argument('--no-captions', action='store_true', help='Disable captions')
    process_parser.add_argument('--no-smart-framing', action='store_true', help='Disable face tracking')
    process_parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')
    process_parser.add_argument('--config', '-c', type=str, help='Path to config file')

    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview highlights without full processing')
    preview_parser.add_argument('input', type=str, help='Input video file')
    preview_parser.add_argument('--quick', '-q', action='store_true', default=True,
                               help='Use fast settings for preview')

    # Clip command (manual clipping)
    clip_parser = subparsers.add_parser('clip', help='Create a manual clip')
    clip_parser.add_argument('input', type=str, help='Input video file')
    clip_parser.add_argument('--start', '-s', type=float, required=True, help='Start time (seconds)')
    clip_parser.add_argument('--end', '-e', type=float, required=True, help='End time (seconds)')
    clip_parser.add_argument('--output', '-o', type=str, help='Output file path')
    clip_parser.add_argument('--no-captions', action='store_true', help='Disable captions')

    # Transcribe command
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe video only')
    transcribe_parser.add_argument('input', type=str, help='Input video/audio file')
    transcribe_parser.add_argument('--output', '-o', type=str, help='Output file path')
    transcribe_parser.add_argument('--format', '-f', type=str, choices=['json', 'srt', 'vtt', 'txt'],
                                  default='json', help='Output format')
    transcribe_parser.add_argument('--model', '-m', type=str,
                                  choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                                  default='base', help='Whisper model size')
    transcribe_parser.add_argument('--language', '-l', type=str, help='Language code')

    # Config command
    config_parser = subparsers.add_parser('config', help='Generate or show configuration')
    config_parser.add_argument('--generate', '-g', type=str, help='Generate config file at path')
    config_parser.add_argument('--preset', type=str, choices=['fast', 'balanced', 'quality', 'max_quality'],
                              default='balanced', help='Preset to use')

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--version', action='version', version='Shorts Bot 1.0.0')

    return parser


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from command line arguments."""
    # Start with preset or default (use get_preset_config for proper GPU detection)
    if hasattr(args, 'preset') and args.preset:
        config = get_preset_config(args.preset)
    elif hasattr(args, 'config') and args.config:
        config = Config.from_file(Path(args.config))
    else:
        config = Config()

    # Apply command line overrides
    if hasattr(args, 'output') and args.output:
        config.output_dir = Path(args.output)

    if hasattr(args, 'platform') and args.platform:
        config.update_for_platform(Platform(args.platform))

    if hasattr(args, 'clips') and args.clips:
        config.highlight.target_clips = args.clips

    if hasattr(args, 'min_duration') and args.min_duration:
        config.highlight.min_clip_duration = args.min_duration

    if hasattr(args, 'max_duration') and args.max_duration:
        config.highlight.max_clip_duration = args.max_duration

    if hasattr(args, 'model') and args.model:
        config.transcription.model = WhisperModel(args.model)

    if hasattr(args, 'language') and args.language:
        config.transcription.language = args.language

    if hasattr(args, 'caption_style') and args.caption_style:
        config.caption.style = CaptionStyle(args.caption_style)

    if hasattr(args, 'no_smart_framing') and args.no_smart_framing:
        config.clipping.smart_framing = False
        config.clipping.face_detection = False

    if hasattr(args, 'keep_temp') and args.keep_temp:
        config.keep_temp_files = True

    return config


def progress_callback(stage: PipelineStage, progress: float, message: str):
    """Print progress updates."""
    bar_width = 30
    filled = int(bar_width * progress)
    bar = '=' * filled + '-' * (bar_width - filled)
    print(f"\r[{stage.value:20s}] [{bar}] {progress*100:5.1f}% {message[:40]:40s}", end='', flush=True)
    if progress >= 1.0:
        print()


def cmd_process(args: argparse.Namespace) -> int:
    """Handle the process command."""
    config = build_config(args)
    pipeline = Pipeline(config, progress_callback=progress_callback if not args.quiet else None)

    input_path = Path(args.input)

    try:
        if args.batch or input_path.is_dir():
            # Batch processing
            result = pipeline.process_directory(input_path, recursive=args.recursive)
            print(f"\n{'='*60}")
            print(f"Batch Processing Complete")
            print(f"{'='*60}")
            print(f"Total videos: {len(result.results)}")
            print(f"Successful:   {result.successful}")
            print(f"Failed:       {result.failed}")
            print(f"Total time:   {result.total_time:.1f}s")
            return 0 if result.failed == 0 else 1
        else:
            # Single file processing
            if not input_path.exists():
                print(f"Error: File not found: {input_path}")
                return 1

            result = pipeline.process_file(input_path)

            print(f"\n{'='*60}")
            print(f"Processing Complete")
            print(f"{'='*60}")

            if result.success:
                print(f"Source:       {result.input_file.path.name}")
                print(f"Duration:     {result.input_file.duration:.1f}s")
                print(f"Clips:        {len(result.clips)}")
                print(f"Output:       {result.output_dir}")
                print(f"Time:         {result.processing_time:.1f}s")
                print(f"\nGenerated clips:")
                for i, clip in enumerate(result.clips, 1):
                    print(f"  {i}. {clip.path.name} ({clip.duration:.1f}s)")
                return 0
            else:
                print(f"Error: {result.error}")
                return 1

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130
    finally:
        if not args.keep_temp:
            pipeline.cleanup()


def cmd_preview(args: argparse.Namespace) -> int:
    """Handle the preview command."""
    config = Config()
    pipeline = Pipeline(config)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    print(f"Analyzing: {input_path.name}")
    print("This may take a moment...\n")

    try:
        highlights = pipeline.preview_highlights(input_path, quick_mode=args.quick)

        print(f"{'='*60}")
        print(f"Found {len(highlights)} potential highlights")
        print(f"{'='*60}\n")

        for i, h in enumerate(highlights, 1):
            print(f"{i}. [{h.start:.1f}s - {h.end:.1f}s] ({h.duration:.1f}s)")
            print(f"   Score: {h.score:.2f}")
            print(f"   Reasons: {', '.join(r.value for r in h.reasons)}")
            print(f"   Preview: \"{h.text[:100]}...\"" if len(h.text) > 100 else f"   Text: \"{h.text}\"")
            print()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        pipeline.cleanup()


def cmd_clip(args: argparse.Namespace) -> int:
    """Handle the clip command."""
    config = Config()
    pipeline = Pipeline(config)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else None

    print(f"Creating clip from {args.start:.1f}s to {args.end:.1f}s")

    try:
        clip = pipeline.create_single_clip(
            file_path=input_path,
            start_time=args.start,
            end_time=args.end,
            output_path=output_path,
            add_captions=not args.no_captions
        )

        print(f"\nClip created: {clip.path}")
        print(f"Duration: {clip.duration:.1f}s")
        print(f"Size: {clip.file_size // 1024} KB")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        pipeline.cleanup()


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle the transcribe command."""
    from src.modules.transcriber import Transcriber
    from src.modules.input_handler import InputHandler

    config = Config()
    if args.model:
        config.transcription.model = WhisperModel(args.model)
    if args.language:
        config.transcription.language = args.language

    input_handler = InputHandler(config)
    transcriber = Transcriber(config)

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    print(f"Transcribing: {input_path.name}")
    print(f"Model: {config.transcription.model.value}")

    try:
        input_file = input_handler.process_file(input_path)
        transcript = transcriber.transcribe(input_file.audio_path)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(f'.{args.format}')

        # Export in requested format
        if args.format == 'json':
            transcript.save(output_path)
        elif args.format == 'srt':
            transcriber.export_srt(transcript, output_path)
        elif args.format == 'vtt':
            transcriber.export_vtt(transcript, output_path)
        elif args.format == 'txt':
            with open(output_path, 'w') as f:
                f.write(transcript.text)

        print(f"\nTranscription complete!")
        print(f"Words: {transcript.word_count}")
        print(f"Duration: {transcript.duration:.1f}s")
        print(f"Language: {transcript.language}")
        print(f"Output: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        input_handler.cleanup()
        transcriber.cleanup()


def cmd_config(args: argparse.Namespace) -> int:
    """Handle the config command."""
    if args.generate:
        try:
            config = get_preset_config(args.preset)
        except ValueError:
            config = Config()
        output_path = Path(args.generate)
        config.save(output_path)
        print(f"Configuration saved to: {output_path}")
        return 0
    else:
        # Show current default config
        try:
            config = get_preset_config(args.preset)
        except ValueError:
            config = Config()
        import json
        print(json.dumps(config.to_dict(), indent=2))
        return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    setup_global_logging(level=log_level)

    # Route to appropriate command handler
    commands = {
        'process': cmd_process,
        'preview': cmd_preview,
        'clip': cmd_clip,
        'transcribe': cmd_transcribe,
        'config': cmd_config,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
