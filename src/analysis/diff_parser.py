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
        """Parse a file's patch into structured hunks."""
        if not file_change.patch:
            return None
        return self.parse_patch(file_change.filename, file_change.patch)

    def parse_patch(self, filename: str, patch: str) -> ParsedDiff:
        """Parse a unified diff patch string into structured hunks and lines."""
        parsed = ParsedDiff(filename=filename)
        lines = patch.split("\n")

        current_hunk: Optional[DiffHunk] = None
        new_line_num = 0

        for line in lines:
            header_match = HUNK_HEADER_PATTERN.match(line)
            if header_match:
                if current_hunk:
                    parsed.hunks.append(current_hunk)

                current_hunk = DiffHunk(
                    old_start=int(header_match.group(1)),
                    old_count=int(header_match.group(2) or 1),
                    new_start=int(header_match.group(3)),
                    new_count=int(header_match.group(4) or 1),
                    header=header_match.group(5).strip(),
                )
                new_line_num = current_hunk.new_start
                continue

            if current_hunk is None:
                continue

            if line.startswith("+"):
                current_hunk.lines.append(HunkLine(
                    line_number=new_line_num, content=line[1:], change_type="added",
                ))
                new_line_num += 1
            elif line.startswith("-"):
                current_hunk.lines.append(HunkLine(
                    line_number=new_line_num, content=line[1:], change_type="removed",
                ))
            elif line.startswith("\\"):
                continue  # "\ No newline at end of file"
            else:
                content = line[1:] if line.startswith(" ") else line
                current_hunk.lines.append(HunkLine(
                    line_number=new_line_num, content=content, change_type="context",
                ))
                new_line_num += 1

        if current_hunk:
            parsed.hunks.append(current_hunk)

        logger.debug(f"Parsed {len(parsed.hunks)} hunks for {filename}")
        return parsed

    def get_changed_line_numbers(self, parsed_diff: ParsedDiff) -> list[int]:
        """Get line numbers of all added lines."""
        return [
            line.line_number
            for hunk in parsed_diff.hunks
            for line in hunk.lines
            if line.change_type == "added"
        ]

    def get_added_code_chunks(self, parsed_diff: ParsedDiff, min_chunk_size: int = 1) -> list[dict]:
        """Get contiguous chunks of added code."""
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
