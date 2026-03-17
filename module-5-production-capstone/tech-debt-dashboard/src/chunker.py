"""Code chunking utilities — reused from Module 4 RAG system."""
import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    metadata: Dict
    chunk_id: str


class CodeChunker:
    """Chunk code files intelligently by language structure."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_file(self, content: str, filename: str, language: str = None) -> List[CodeChunk]:
        if language is None:
            language = self._detect_language(filename)

        if language == "python":
            return self._chunk_python(content, filename)
        elif language in ("javascript", "typescript"):
            return self._chunk_javascript(content, filename, language)
        else:
            return self._chunk_generic(content, filename, language)

    def _chunk_python(self, content: str, filename: str) -> List[CodeChunk]:
        chunks = []
        pattern = r'((?:^@\w+.*\n)*^(?:def|class|async def)\s+\w+[^:]*:.*?)(?=\n(?:@|\s*def|\s*class|\s*async def)|\Z)'
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))

        if matches:
            first_match_start = matches[0].start()
            if first_match_start > 0:
                header = content[:first_match_start].strip()
                if header:
                    chunks.append(CodeChunk(
                        content=header,
                        metadata={"filename": filename, "language": "python", "type": "header", "line_start": 1},
                        chunk_id=f"{filename}:header"
                    ))

            for i, match in enumerate(matches):
                chunk_content = match.group(1).strip()
                line_start = content[:match.start()].count('\n') + 1
                name_match = re.search(r'(?:def|class|async def)\s+(\w+)', chunk_content)
                name = name_match.group(1) if name_match else f"block_{i}"
                line_count = chunk_content.count('\n') + 1

                chunks.append(CodeChunk(
                    content=chunk_content,
                    metadata={
                        "filename": filename,
                        "language": "python",
                        "type": "class" if "class " in chunk_content else "function",
                        "name": name,
                        "line_start": line_start,
                        "line_count": line_count,
                    },
                    chunk_id=f"{filename}:{name}"
                ))
        else:
            return self._chunk_generic(content, filename, "python")

        return chunks

    def _chunk_javascript(self, content: str, filename: str, language: str) -> List[CodeChunk]:
        chunks = []
        pattern = r'((?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+\w+.*?)(?=\n(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+|\Z)'
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))

        if matches:
            first_match_start = matches[0].start()
            if first_match_start > 0:
                header = content[:first_match_start].strip()
                if header:
                    chunks.append(CodeChunk(
                        content=header,
                        metadata={"filename": filename, "language": language, "type": "header", "line_start": 1},
                        chunk_id=f"{filename}:header"
                    ))

            for i, match in enumerate(matches):
                chunk_content = match.group(1).strip()
                line_start = content[:match.start()].count('\n') + 1
                name_match = re.search(r'(?:function|class|const|let|var)\s+(\w+)', chunk_content)
                name = name_match.group(1) if name_match else f"block_{i}"
                line_count = chunk_content.count('\n') + 1

                chunks.append(CodeChunk(
                    content=chunk_content,
                    metadata={
                        "filename": filename,
                        "language": language,
                        "type": "class" if "class " in chunk_content else "function",
                        "name": name,
                        "line_start": line_start,
                        "line_count": line_count,
                    },
                    chunk_id=f"{filename}:{name}"
                ))
        else:
            return self._chunk_generic(content, filename, language)

        return chunks

    def _chunk_generic(self, content: str, filename: str, language: str) -> List[CodeChunk]:
        chunks = []
        lines = content.split('\n')
        total_chars = 0
        current_chunk: List[str] = []
        chunk_num = 0

        for i, line in enumerate(lines):
            current_chunk.append(line)
            total_chars += len(line)

            if total_chars >= self.chunk_size:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    metadata={
                        "filename": filename,
                        "language": language,
                        "type": "chunk",
                        "line_start": i - len(current_chunk) + 1,
                        "line_count": len(current_chunk),
                    },
                    chunk_id=f"{filename}:chunk_{chunk_num}"
                ))
                overlap_chars = 0
                overlap_lines: List[str] = []
                for line in reversed(current_chunk):
                    if overlap_chars >= self.chunk_overlap:
                        break
                    overlap_lines.insert(0, line)
                    overlap_chars += len(line)
                current_chunk = overlap_lines
                total_chars = overlap_chars
                chunk_num += 1

        if current_chunk:
            chunks.append(CodeChunk(
                content='\n'.join(current_chunk),
                metadata={
                    "filename": filename,
                    "language": language,
                    "type": "chunk",
                    "line_start": len(lines) - len(current_chunk) + 1,
                    "line_count": len(current_chunk),
                },
                chunk_id=f"{filename}:chunk_{chunk_num}"
            ))

        return chunks

    def _detect_language(self, filename: str) -> str:
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
            '.go': 'go', '.rs': 'rust', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
        }
        for ext, lang in ext_map.items():
            if filename.endswith(ext):
                return lang
        return 'unknown'
