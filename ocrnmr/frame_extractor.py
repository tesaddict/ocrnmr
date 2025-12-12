"""Frame extraction from video files using ffmpeg."""

import subprocess
import logging
import time
from pathlib import Path
from typing import Tuple, Optional, List, Generator
from ocrnmr.profiler import profiler

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

logger = logging.getLogger(__name__)

def _parse_mjpeg_frames_from_buffer(
    buffer: bytearray,
    interval_seconds: float,
    in_frame: bool,
    frame_start: int,
    start_time: Optional[float] = None,
    frame_count: int = 0
) -> Tuple[bytearray, bool, int, List[Tuple[float, bytes]]]:
    """
    Parse MJPEG frames from a buffer incrementally.
    Returns (remaining_buffer, in_frame, frame_start, list_of_new_frames)
    """
    new_frames = []
    i = 0
    while i < len(buffer) - 1:
        # Look for JPEG start marker (0xFF 0xD8)
        if not in_frame and buffer[i] == 0xFF and buffer[i + 1] == 0xD8:
            in_frame = True
            frame_start = i
            i += 2
            continue
        
        # Look for JPEG end marker (0xFF 0xD9)
        if in_frame and buffer[i] == 0xFF and buffer[i + 1] == 0xD9:
            # Complete frame found
            frame_end = i + 2
            frame_bytes = bytes(buffer[frame_start:frame_end])
            
            # Calculate timestamp based on frame index
            timestamp = (frame_count * interval_seconds) + (start_time if start_time else 0)
            new_frames.append((timestamp, frame_bytes))
            frame_count += 1
            
            # Remove processed frame data from buffer
            buffer = buffer[frame_end:]
            in_frame = False
            i = 0  # Reset index after removing data
            continue
        
        i += 1
    
    return buffer, in_frame, frame_start, new_frames


def extract_frames_generator(
    media_file: Path,
    interval_seconds: float = 2.0,
    max_dimension: Optional[int] = 800,
    duration: Optional[float] = None,
    hwaccel: Optional[str] = None,
    start_time: Optional[float] = None,
    timestamps: Optional[List[float]] = None
) -> Generator[Tuple[float, bytes], None, None]:
    """
    Generator that yields frames from a video file using ffmpeg and in-memory piping.
    
    Args:
        media_file: Path to the video file
        interval_seconds: Interval between frames in seconds (default: 2.0)
        max_dimension: Maximum width or height in pixels.
        duration: Maximum duration to extract in seconds.
        hwaccel: Hardware acceleration for decoding.
        start_time: Start time in seconds for frame extraction.
        timestamps: Optional list of specific timestamps to extract.
    
    Yields:
        Tuple of (timestamp, frame_bytes)
    """
    profiler.start_timer("ffmpeg_extract_generator")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python is not installed. Please install it with: pip install ffmpeg-python")
    
    if not media_file.exists():
        raise FileNotFoundError(f"Media file not found: {media_file}")
    
    # Calculate fps for extraction
    fps = 1.0 / interval_seconds
    
    # Build input args
    input_kwargs = {
        'threads': '0',  # Use all available threads
    }
    
    if hwaccel:
        input_kwargs['hwaccel'] = hwaccel
        if hwaccel == 'videotoolbox':
            input_kwargs['hwaccel_output_format'] = 'nv12'
            
    # Apply start time and duration
    if timestamps is None:
        if start_time is not None:
            input_kwargs['ss'] = start_time
        elif duration is not None:
            input_kwargs['ss'] = 0
        
        if duration is not None:
            input_kwargs['t'] = duration
            
    try:
        input_stream = ffmpeg.input(str(media_file), **input_kwargs)
        
        # Apply filters
        if timestamps is not None:
            # Extract specific timestamps
            select_expr = '+'.join([f"between(t,{t-0.1},{t+0.1})" for t in timestamps])
            
            if 'skip_frame' in input_kwargs:
                del input_kwargs['skip_frame']
            
            input_stream = ffmpeg.input(str(media_file), **input_kwargs)
            
            if max_dimension is not None and max_dimension > 0:
                filtered_stream = input_stream.filter('scale', max_dimension, -1, flags='neighbor').filter('select', select_expr)
            else:
                filtered_stream = input_stream.filter('select', select_expr)
                
            output_kwargs = {'vsync': 'vfr'}
        else:
            # Regular interval extraction
            if max_dimension is not None and max_dimension > 0:
                filtered_stream = input_stream.filter('scale', max_dimension, -1, flags='neighbor').filter('fps', fps)
            else:
                filtered_stream = input_stream.filter('fps', fps)
            output_kwargs = {}

        # Output settings for MJPEG pipe
        output_kwargs.update({
            'format': 'image2pipe',
            'vcodec': 'mjpeg',
            'q:v': '31',
        })
        
        # Run ffmpeg with pipe output
        process = (
            filtered_stream
            .output('pipe:', **output_kwargs)
            .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
        )
        
        # Read from pipe incrementally
        buffer = bytearray()
        in_frame = False
        frame_start = 0
        frame_count = 0
        chunk_size = 65536  # 64KB chunks
        
        try:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                    
                buffer.extend(chunk)
                
                # Parse frames from buffer
                profiler.start_timer("parse_mjpeg")
                buffer, in_frame, frame_start, new_frames = _parse_mjpeg_frames_from_buffer(
                    buffer, interval_seconds, in_frame, frame_start, start_time, frame_count
                )
                profiler.stop_timer("parse_mjpeg")
                
                for frame in new_frames:
                    frame_count += 1
                    yield frame
                    
            # Wait for process to finish
            process.wait()
            
            profiler.stop_timer("ffmpeg_extract_generator")
            
            if process.returncode != 0:
                # Check stderr for errors
                stderr = process.stderr.read().decode('utf8', errors='ignore')
                # Ignore SIGPIPE/SIGTERM if we stopped reading early
                if process.returncode not in (-15, -13, 255):
                    logger.warning(f"FFmpeg exited with code {process.returncode}: {stderr[-200:]}")

        except GeneratorExit:
            # Generator was closed explicitly
            if process.poll() is None:
                process.terminate()
            raise
        except Exception:
            # Other exceptions
            if process.poll() is None:
                process.terminate()
            raise
        finally:
            # Ensure cleanup
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=1.0)
                except Exception:
                    process.kill()
            
            # Close pipes
            if process.stdout: process.stdout.close()
            if process.stderr: process.stderr.close()
            
    except Exception as e:
        logger.error(f"Error in frame extraction generator: {e}")
        raise


def extract_frames_batch(
    media_file: Path,
    interval_seconds: float = 2.0,
    max_dimension: Optional[int] = 800,
    duration: Optional[float] = None,
    hwaccel: Optional[str] = None,
    start_time: Optional[float] = None,
    timestamps: Optional[List[float]] = None
) -> List[Tuple[float, bytes]]:
    """
    Legacy wrapper for extract_frames_generator to return a list.
    """
    return list(extract_frames_generator(
        media_file, interval_seconds, max_dimension, duration, hwaccel, start_time, timestamps
    ))
