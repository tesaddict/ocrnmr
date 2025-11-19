"""Frame extraction from video files using ffmpeg."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

# Track active ffmpeg processes for cleanup
_active_processes = []


# Don't register a global signal handler - it interferes with rich display
# Signal handling is done at the application level instead
# The FrameCache.shutdown() method will kill FFMPEG processes when needed


def _parse_mjpeg_frames_from_buffer(
    buffer: bytearray,
    frames: List[Tuple[float, bytes]],
    interval_seconds: float,
    in_frame: bool,
    frame_start: int,
    start_time: Optional[float] = None
) -> Tuple[bytearray, bool, int]:
    """
    Parse MJPEG frames from a buffer incrementally.
    
    Args:
        buffer: Buffer containing MJPEG data
        frames: List to append complete frames to
        interval_seconds: Interval between frames for timestamp calculation
        in_frame: Whether we're currently inside a frame
        frame_start: Start position of current frame in buffer
    
    Returns:
        Tuple of (remaining_buffer, in_frame, frame_start)
    """
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
            timestamp = (len(frames) * interval_seconds) + (start_time if start_time else 0)
            frames.append((timestamp, frame_bytes))
            
            # Remove processed frame data from buffer
            buffer = buffer[frame_end:]
            in_frame = False
            i = 0  # Reset index after removing data
            continue
        
        i += 1
    
    return buffer, in_frame, frame_start


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
    Extract all frames from a video file in a single ffmpeg pass (batch extraction).
    
    This is much faster than individual frame extraction because it:
    - Uses a single ffmpeg process instead of hundreds
    - Avoids repeated seeks
    - Pipes frames directly to memory
    - Uses optimized software encoding for maximum speed
    
    Args:
        media_file: Path to the video file
        interval_seconds: Interval between frames in seconds (default: 2.0)
        max_dimension: Maximum width or height in pixels. If set, frames are scaled by ffmpeg.
                      Default: 800px for optimal OCR performance
        duration: Maximum duration to extract in seconds. If None, extracts entire video.
                  Useful for testing (e.g., duration=60 for first minute)
        hwaccel: Hardware acceleration for decoding (None for software, 'videotoolbox' for macOS,
                 'vaapi' for Linux, 'd3d11va'/'dxva2' for Windows). Significantly speeds up decode time.
        start_time: Start time in seconds for frame extraction (default: 0). If specified, extraction
                    starts from this timestamp. Useful for focusing on a specific region of the video.
        timestamps: Optional list of specific timestamps to extract. If provided, extracts ONLY these frames.
                    This uses a complex filter chain and may be slower per frame but faster for sparse extraction.
                    Overrides interval_seconds, duration, and start_time.
    
    Returns:
        List of tuples (timestamp, frame_bytes) where timestamp is in seconds (relative to start_time)
    
    Raises:
        RuntimeError: If ffmpeg is not available or video cannot be processed
    """
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python is not installed. Please install it with: pip install ffmpeg-python")
    
    if not media_file.exists():
        raise FileNotFoundError(f"Media file not found: {media_file}")
    
    try:
        # Probe the video to get duration
        probe = ffmpeg.probe(str(media_file))
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        if video_stream is None:
            raise RuntimeError(f"No video stream found in {media_file}")
        
        video_duration = float(probe['format'].get('duration', 0))
        if video_duration <= 0:
            raise RuntimeError(f"Invalid video duration: {video_duration}")
        
        # Calculate fps for extraction (e.g., interval_seconds=2.0 means fps=0.5)
        fps = 1.0 / interval_seconds
        
        # Build ffmpeg pipeline for batch extraction
        # Optimized for maximum decode speed with hardware acceleration
        input_kwargs = {
            'skip_frame': 'nokey',  # Skip non-keyframes - much faster for sparse frame extraction
            'threads': '0',  # Use all available threads for decoding
        }
        
        # Enable hardware acceleration for decoding (if configured)
        # This significantly speeds up decode time on supported systems
        # Note: We still use software encoding for MJPEG since hardware MJPEG encoding is rarely supported
        if hwaccel:
            input_kwargs['hwaccel'] = hwaccel
            # Set appropriate output format for VideoToolbox (macOS)
            if hwaccel == 'videotoolbox':
                input_kwargs['hwaccel_output_format'] = 'nv12'  # Native format for VideoToolbox
        
        # Apply duration limit and seek position as INPUT options
        # Using -ss before -i is faster than seeking after input
        if timestamps is None:
            if start_time is not None:
                input_kwargs['ss'] = start_time  # Start from specified time
            elif duration is not None:
                input_kwargs['ss'] = 0  # Start from beginning if duration is set
            
            if duration is not None:
                input_kwargs['t'] = duration
        
        input_stream = ffmpeg.input(str(media_file), **input_kwargs)
        
        # Apply filters: scale first (if needed), then fps filter to extract at specified interval
        # Use nearest neighbor scaling for maximum speed (quality doesn't matter for OCR)
        if timestamps is not None:
            # Extract specific timestamps using select filter
            # select='eq(t,T1)+eq(t,T2)+...'
            select_expr = '+'.join([f"between(t,{t-0.1},{t+0.1})" for t in timestamps])
            # print(f"DEBUG: select_expr={select_expr}")
            # Use a slightly wider window (0.1s) to ensure we catch a frame near the timestamp
            # Then force fps to match the number of timestamps to avoid duplicates
            
            # For specific timestamps, we can't use skip_frame=nokey as we need precise frames
            # So we remove it from input_kwargs if it was added
            if 'skip_frame' in input_kwargs:
                del input_kwargs['skip_frame']
            
            # Re-create input stream with updated kwargs
            input_stream = ffmpeg.input(str(media_file), **input_kwargs)
            
            if max_dimension is not None and max_dimension > 0:
                filtered_stream = input_stream.filter('scale', max_dimension, -1, flags='neighbor').filter('select', select_expr)
            else:
                filtered_stream = input_stream.filter('select', select_expr)
        else:
            if max_dimension is not None and max_dimension > 0:
                filtered_stream = input_stream.filter('scale', max_dimension, -1, flags='neighbor').filter('fps', fps)
            else:
                filtered_stream = input_stream.filter('fps', fps)
        
        # Output to pipe as MJPEG stream (image2pipe format)
        # Optimized for MAXIMUM SPEED: lowest quality, fastest encoding
        output_kwargs = {
            'format': 'image2pipe',
            'vcodec': 'mjpeg',
            'fps_mode': 'passthrough',  # Don't duplicate/drop frames
            'an': None,  # Disable audio processing
            'q:v': '31',  # Lowest quality (1-31, 31 = fastest encoding, quality doesn't matter)
            'threads': '0',  # Use all available threads for encoding
            'fflags': '+fastseek',  # Faster seeking
        }
        
        # Use PIPE for stderr to capture error messages
        stderr_handle = subprocess.PIPE
        
        process = (
            filtered_stream
            .output('pipe:', **output_kwargs)
            .run_async(pipe_stdout=True, pipe_stderr=stderr_handle, quiet=True)
        )
        
        # Track process for cleanup
        _active_processes.append(process)
        
        # Give FFMPEG a moment to start up and begin producing output
        # This helps avoid hanging on the first read if FFMPEG is slow to initialize
        time.sleep(0.1)  # Small delay to let FFMPEG start
        
        try:
            # Calculate expected number of frames based on duration
            # If duration is specified, we should get approximately duration/interval_seconds frames
            # Add a small buffer (e.g., +1) to account for timing variations
            expected_frames = None
            if timestamps is not None:
                expected_frames = len(timestamps)
            elif duration is not None:
                expected_frames = int(duration / interval_seconds) + 1
            
            # Read stdout incrementally in chunks to avoid blocking
            # This ensures efficient processing of frame data as it arrives
            frames = []
            buffer = bytearray()
            chunk_size = 65536  # 64KB chunks
            max_timeout = 300  # Maximum total timeout in seconds
            read_timeout = 1.0  # Timeout per read operation in seconds
            
            start_time_perf = time.time()
            
            # Parse MJPEG frames incrementally as data arrives
            # MJPEG frames are separated by JPEG markers:
            # Start: 0xFF 0xD8
            # End: 0xFF 0xD9
            in_frame = False
            frame_start = 0
            
            while True:
                # Check for overall timeout
                if time.time() - start_time_perf > max_timeout:
                    raise subprocess.TimeoutExpired(process.args, max_timeout)
                
                # If we have enough frames for the duration, stop reading early
                # This prevents reading the entire video when duration limit isn't respected
                # Use a small buffer (e.g., +2) to account for frames that might arrive slightly late
                if expected_frames is not None and len(frames) >= expected_frames + 2:
                    # We have enough frames (with buffer), kill the process to stop it from continuing
                    try:
                        process.kill()
                    except Exception:
                        pass
                    # Read any remaining buffered data
                    try:
                        remaining_data = process.stdout.read()
                        if remaining_data:
                            buffer.extend(remaining_data)
                            # Parse any complete frames from remaining data
                            buffer, in_frame, frame_start = _parse_mjpeg_frames_from_buffer(
                                buffer, frames, interval_seconds, in_frame, frame_start, start_time
                            )
                    except Exception:
                        pass
                    # Trim to expected number of frames
                    if len(frames) > expected_frames:
                        frames = frames[:expected_frames]
                    break
                
                # Check if process has finished
                if process.poll() is not None:
                    # Process finished, read any remaining data
                    remaining_data = process.stdout.read()
                    if remaining_data:
                        buffer.extend(remaining_data)
                        # Parse frames from the remaining data before breaking
                        buffer, in_frame, frame_start = _parse_mjpeg_frames_from_buffer(
                            buffer, frames, interval_seconds, in_frame, frame_start, start_time
                        )
                    break
                
                # Try to read a chunk with timeout
                # Use select for non-blocking read if available, otherwise use a small timeout
                try:
                    import select
                    # Check if stdout is ready to read (with timeout)
                    ready, _, _ = select.select([process.stdout], [], [], read_timeout)
                    if ready:
                        # Data is available, read it
                        chunk = process.stdout.read(chunk_size)
                        if not chunk:
                            # EOF reached
                            break
                        buffer.extend(chunk)
                    else:
                        # Timeout on read - check if process is still running
                        if process.poll() is not None:
                            # Process finished while we were waiting
                            remaining_data = process.stdout.read()
                            if remaining_data:
                                buffer.extend(remaining_data)
                                # Parse frames from the remaining data before breaking
                                buffer, in_frame, frame_start = _parse_mjpeg_frames_from_buffer(
                                    buffer, frames, interval_seconds, in_frame, frame_start, start_time
                                )
                            break
                        # Process still running but no data yet - continue loop
                        # This handles the case where FFMPEG is still initializing
                        continue
                except (ImportError, OSError):
                    # select not available or failed (Windows/macOS in some cases)
                    # Check if process is still running before blocking read
                    if process.poll() is None:
                        # Process is running, try to read (this may block briefly)
                        # On some systems, we can't avoid blocking, but we've already
                        # waited a bit for FFMPEG to start
                        chunk = process.stdout.read(chunk_size)
                        if not chunk:
                            # EOF reached
                            break
                        buffer.extend(chunk)
                    else:
                        # Process finished, read remaining data
                        remaining_data = process.stdout.read()
                        if remaining_data:
                            buffer.extend(remaining_data)
                            # Parse frames from the remaining data
                            buffer, in_frame, frame_start = _parse_mjpeg_frames_from_buffer(
                                buffer, frames, interval_seconds, in_frame, frame_start, start_time
                            )
                        break
                
                # Parse frames from buffer as data arrives
                buffer, in_frame, frame_start = _parse_mjpeg_frames_from_buffer(
                    buffer, frames, interval_seconds, in_frame, frame_start, start_time
                )
            
            # Wait for process to complete
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Process didn't finish in time, kill it
                process.kill()
                process.wait()
            
            # Remove from tracking after completion
            if process in _active_processes:
                _active_processes.remove(process)
            
            # Check for errors (but ignore SIGTERM/SIGINT kills - return code 255 or -15)
            # These are expected if the process was interrupted
            if process.returncode != 0 and process.returncode not in (255, -15, -2):
                # Try to read stderr if available (software mode)
                stderr_data = None
                if stderr_handle == subprocess.PIPE and process.stderr:
                    try:
                        stderr_data = process.stderr.read()
                    except Exception:
                        pass
                
                if stderr_data:
                    stderr_text = stderr_data.decode('utf-8', errors='ignore')
                    # Check if stderr indicates it was killed by signal
                    if 'signal' in stderr_text.lower() or 'killed' in stderr_text.lower():
                        raise RuntimeError(f"FFmpeg process was terminated (return code {process.returncode})")
                    raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {stderr_text[-500:]}")
                else:
                    # Hardware mode - stderr was redirected, can't check it
                    raise RuntimeError(f"FFmpeg failed with return code {process.returncode} (stderr not available)")
            
            # Handle any remaining frame data in buffer (last frame might not have end marker)
            if in_frame and len(buffer) > frame_start:
                # Check if it looks like a valid JPEG
                if buffer[frame_start] == 0xFF and buffer[frame_start + 1] == 0xD8:
                    timestamp = (len(frames) * interval_seconds) + (start_time if start_time else 0)
                    frames.append((timestamp, bytes(buffer[frame_start:])))
            
            # If using timestamps, fix up the returned timestamps to match requested ones
            if timestamps is not None and len(frames) > 0:
                # Map extracted frames to requested timestamps
                # This is approximate since we can't guarantee exact frame alignment
                mapped_frames = []
                for i, (_, frame_data) in enumerate(frames):
                    if i < len(timestamps):
                        mapped_frames.append((timestamps[i], frame_data))
                return mapped_frames
            
            return frames
            
        except KeyboardInterrupt:
            # User interrupted - kill process immediately and re-raise
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass
            if process in _active_processes:
                _active_processes.remove(process)
            raise
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except Exception:
                pass
            if process in _active_processes:
                _active_processes.remove(process)
            raise RuntimeError(f"Frame extraction timed out for {media_file}")
        except Exception as e:
            # Any other error - ensure process is killed
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass
            if process in _active_processes:
                _active_processes.remove(process)
            raise RuntimeError(f"Error reading frames from pipe: {str(e)}")
            
    except Exception as e:
        raise RuntimeError(f"Error extracting frames from {media_file}: {str(e)}")
