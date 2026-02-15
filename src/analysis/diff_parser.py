"""Parser for GitHub diff patches."""

import re
import logging
from typing import Optional

from src.github.models import FileChange, DiffHunk, HunkLine, ParsedDiff

logger = logging.getLogger(__name__)

# Regex to match hunk headers like @@ -1,5 +1,7 @@
HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


class DiffParser:
    """Parser for unified diff format."""
    
    def parse_file_diff(self, file_change: FileChange) -> Optional[ParsedDiff]:
        """
        Parse a file's patch into structured hunks.
        
        Args:
            file_change: FileChange object containing the patch.
            
        Returns:
            ParsedDiff object or None if no patch.
        """
        if not file_change.patch:
            return None
        
        return self.parse_patch(file_change.filename, file_change.patch)
    
    def parse_patch(self, filename: str, patch: str) -> ParsedDiff:
        """
        Parse a unified diff patch string.
        
        Args:
            filename: Name of the file.
            patch: Unified diff patch string.
            
        Returns:
            ParsedDiff object with parsed hunks.
        """
        parsed = ParsedDiff(filename=filename)
        lines = patch.split("\n")
        
        current_hunk: Optional[DiffHunk] = None
        new_line_num = 0
        
        for line in lines:
            # Check for hunk header
            header_match = HUNK_HEADER_PATTERN.match(line)
            if header_match:
                # Save previous hunk if exists
                if current_hunk:
                    parsed.hunks.append(current_hunk)
                
                # Parse hunk header
                old_start = int(header_match.group(1))
                old_count = int(header_match.group(2) or 1)
                new_start = int(header_match.group(3))
                new_count = int(header_match.group(4) or 1)
                header_context = header_match.group(5).strip()
                
                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=header_context,
                )
                new_line_num = new_start
                continue
            
            # Skip if no current hunk (shouldn't happen with valid diff)
            if current_hunk is None:
                continue
            
            # Parse diff lines
            if line.startswith("+"):
                # Added line
                current_hunk.lines.append(HunkLine(
                    line_number=new_line_num,
                    content=line[1:],  # Remove the + prefix
                    change_type="added",
                ))
                new_line_num += 1
            elif line.startswith("-"):
                # Removed line (don't increment new_line_num)
                current_hunk.lines.append(HunkLine(
                    line_number=new_line_num,  # Use current position for reference
                    content=line[1:],  # Remove the - prefix
                    change_type="removed",
                ))
            elif line.startswith(" ") or line == "":
                # Context line
                content = line[1:] if line.startswith(" ") else line
                current_hunk.lines.append(HunkLine(
                    line_number=new_line_num,
                    content=content,
                    change_type="context",
                ))
                new_line_num += 1
            elif line.startswith("\\"):
                # "\ No newline at end of file" - skip
                continue
        
        # Don't forget the last hunk
        if current_hunk:
            parsed.hunks.append(current_hunk)
        
        logger.debug(f"Parsed {len(parsed.hunks)} hunks for {filename}")
        return parsed
    
    def get_changed_line_numbers(self, parsed_diff: ParsedDiff) -> list[int]:
        """
        Get line numbers of all added/modified lines.
        
        Args:
            parsed_diff: ParsedDiff object.
            
        Returns:
            List of line numbers that were added or modified.
        """
        line_numbers = []
        for hunk in parsed_diff.hunks:
            for line in hunk.lines:
                if line.change_type == "added":
                    line_numbers.append(line.line_number)
        return line_numbers
    
    def get_context_around_line(
        self,
        parsed_diff: ParsedDiff,
        line_number: int,
        context_lines: int = 3,
    ) -> list[HunkLine]:
        """
        Get lines around a specific line number for context.
        
        Args:
            parsed_diff: ParsedDiff object.
            line_number: Target line number.
            context_lines: Number of context lines before and after.
            
        Returns:
            List of HunkLine objects around the target line.
        """
        all_lines = []
        for hunk in parsed_diff.hunks:
            all_lines.extend(hunk.lines)
        
        # Find the target line index
        target_idx = None
        for idx, line in enumerate(all_lines):
            if line.line_number == line_number and line.change_type == "added":
                target_idx = idx
                break
        
        if target_idx is None:
            return []
        
        # Get context
        start = max(0, target_idx - context_lines)
        end = min(len(all_lines), target_idx + context_lines + 1)
        
        return all_lines[start:end]
    
    def extract_code_block(
        self,
        parsed_diff: ParsedDiff,
        start_line: int,
        end_line: int,
    ) -> str:
        """
        Extract code from a range of lines.
        
        Args:
            parsed_diff: ParsedDiff object.
            start_line: Starting line number.
            end_line: Ending line number (inclusive).
            
        Returns:
            Code block as string.
        """
        lines = []
        for hunk in parsed_diff.hunks:
            for line in hunk.lines:
                if start_line <= line.line_number <= end_line:
                    if line.change_type != "removed":
                        lines.append(line.content)
        
        return "\n".join(lines)
    
    def get_added_code_chunks(
        self,
        parsed_diff: ParsedDiff,
        min_chunk_size: int = 1,
    ) -> list[dict]:
        """
        Get contiguous chunks of added code.
        
        Args:
            parsed_diff: ParsedDiff object.
            min_chunk_size: Minimum number of lines for a chunk.
            
        Returns:
            List of dicts with 'start_line', 'end_line', 'code' keys.
        """
        chunks = []
        current_chunk_lines = []
        current_start = None
        
        for hunk in parsed_diff.hunks:
            for line in hunk.lines:
                if line.change_type == "added":
                    if current_start is None:
                        current_start = line.line_number
                    current_chunk_lines.append(line)
                else:
                    # End of added chunk
                    if current_chunk_lines and len(current_chunk_lines) >= min_chunk_size:
                        chunks.append({
                            "start_line": current_start,
                            "end_line": current_chunk_lines[-1].line_number,
                            "code": "\n".join(l.content for l in current_chunk_lines),
                        })
                    current_chunk_lines = []
                    current_start = None
        
        # Don't forget last chunk
        if current_chunk_lines and len(current_chunk_lines) >= min_chunk_size:
            chunks.append({
                "start_line": current_start,
                "end_line": current_chunk_lines[-1].line_number,
                "code": "\n".join(l.content for l in current_chunk_lines),
            })
        
        return chunks
