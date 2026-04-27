"""
Text chunking utilities.
Splits large text into overlapping chunks at word boundaries.
"""

from typing import List, Tuple, Iterator


def chunk_text_by_words(
    text: str,
    chunk_size: int,
    overlap_size: int,
) -> Iterator[Tuple[int, str]]:
    """
    Yield (absolute_offset, chunk_text) tuples.
    
    Chunking is done at character level but snapped to word boundaries
    so tokens are never split across chunks.
    Overlap ensures matches near chunk boundaries are not missed.
    
    Args:
        text:        Full document text.
        chunk_size:  Approximate characters per chunk (before overlap).
        overlap_size: Characters of overlap on each side.
    """
    text_len = len(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_size < 0:
        raise ValueError("overlap_size must be non-negative")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be less than chunk_size")

    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap end to the next word boundary (avoid splitting tokens)
        if end < text_len:
            snap = text.find(' ', end)
            if snap != -1 and snap - end < 200:   # don't wander too far
                end = snap

        # Apply right overlap
        overlap_end = min(end + overlap_size, text_len)

        # Apply left overlap (don't go before 0)
        overlap_start = max(0, start - overlap_size)

        chunk = text[overlap_start:overlap_end]
        yield overlap_start, chunk

        if end >= text_len:
            break
        start = end


def chunk_text_streaming(
    file_path: str,
    chunk_size: int,
    overlap_size: int,
    encoding: str = "utf-8",
) -> Iterator[Tuple[int, int, str]]:
    """
    Stream-chunk a large file without loading it all into memory.
    Yields (chunk_id, absolute_offset, chunk_text).
    
    Uses a rolling buffer for overlap.
    """
    buffer = ""
    abs_offset = 0
    buffer_start = 0   # absolute position of buffer[0]
    chunk_id = 0

    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        while True:
            data = f.read(chunk_size * 2)   # read 2x to have room for snapping
            if not data and not buffer:
                break

            buffer += data

            while len(buffer) >= chunk_size:
                # Find cut point
                cut = chunk_size
                snap = buffer.find(' ', cut)
                if snap != -1 and snap - cut < 200:
                    cut = snap

                # Build chunk with overlap
                left_extra = min(overlap_size, buffer_start - 0)
                # can't expand left past file start, but buffer already contains it
                # because we track buffer_start

                chunk_text = buffer[:cut + overlap_size]
                yield chunk_id, buffer_start, chunk_text

                # Advance: keep overlap tail in buffer
                advance = max(1, cut - overlap_size)
                buffer_start += advance
                buffer = buffer[advance:]
                chunk_id += 1

            if not data:
                # flush remaining buffer
                if buffer:
                    yield chunk_id, buffer_start, buffer
                break


def estimate_chunk_count(file_size_bytes: int, chunk_size: int) -> int:
    """Rough estimate of chunk count for progress display."""
    return max(1, file_size_bytes // chunk_size)
