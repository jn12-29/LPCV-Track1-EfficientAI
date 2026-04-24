# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.scripts.generate_test_summary import (
    extract_file_and_line,
    extract_relevant_stack_trace,
)

"""
Test extract_file_and_line function:
1. File path with colon format - Tests extracting file and line from `file.py:123:` format
2. Traceback format - Tests extracting file and line from `File "file.py", line 123` format
3. Extraction from message - Tests extracting file and line from the message when not in stack trace
4. No match case - Tests behavior when no file and line can be extracted
5. None stack trace- Tests behavior with `None` stack trace
6. Multiple matches - Tests that the last match is returned when multiple matches exist

Test extract_relevant_stack_trace function:
1. Short stack trace - Tests that the whole trace is returned for short stack traces
2. Medium-length stack trace - Tests that the whole trace is returned for medium-length stack traces
3. Long stack trace with error type - Tests that only the relevant part is extracted for long stack traces
4. None stack trace - Tests behavior with `None` stack trace
5. Complex error message - Tests handling of error messages containing multiple colons
"""


def test_extract_file_and_line_colon_format() -> None:
    """Test extracting file and line from 'file.py:123:' format."""
    stack_trace = "some_module.py:42: in function_name"
    file_path, line_number = extract_file_and_line(stack_trace, "Error message")
    assert file_path == "some_module.py"
    assert line_number == "42"


def test_extract_file_and_line_traceback_format() -> None:
    """Test extracting file and line from 'File "file.py", line 123' format."""
    stack_trace = 'File "/path/to/file.py", line 123, in some_function'
    file_path, line_number = extract_file_and_line(stack_trace, "Error message")
    assert file_path == "/path/to/file.py"
    assert line_number == "123"


def test_extract_file_and_line_from_message() -> None:
    """Test extracting file and line from message when not in stack trace."""
    stack_trace = "No file path here"
    message = 'Error in File "/path/to/file.py", line 456'
    file_path, line_number = extract_file_and_line(stack_trace, message)
    assert file_path == "/path/to/file.py"
    assert line_number == "456"


def test_extract_file_and_line_no_match() -> None:
    """Test when no file and line can be extracted."""
    stack_trace = "No file path or line number here"
    file_path, line_number = extract_file_and_line(stack_trace, "Error message")
    assert file_path is None
    assert line_number is None


def test_extract_file_and_line_none_stack_trace() -> None:
    """Test with None stack trace."""
    file_path, line_number = extract_file_and_line(None, "Error message")
    assert file_path is None
    assert line_number is None


def test_extract_file_and_line_multiple_matches() -> None:
    """Test with multiple matches in stack trace - should return the last one."""
    stack_trace = """
    File "/path/to/file1.py", line 10, in function1
    File "/path/to/file2.py", line 20, in function2
    File "/path/to/file3.py", line 30, in function3
    """
    file_path, line_number = extract_file_and_line(stack_trace, "Error message")
    assert file_path == "/path/to/file3.py"
    assert line_number == "30"


def test_extract_relevant_stack_trace_short() -> None:
    """Test with a short stack trace - should return the whole trace."""
    stack_trace = "ValueError: Invalid value"
    message = "ValueError: Invalid value"
    result = extract_relevant_stack_trace(stack_trace, message)
    assert result is not None
    assert result == stack_trace


def test_extract_relevant_stack_trace_medium() -> None:
    """Test with a medium-length stack trace - should return the whole trace."""
    stack_trace = """
    File "/path/to/file1.py", line 10, in function1
    File "/path/to/file2.py", line 20, in function2
    ValueError: Invalid value
    """
    message = "ValueError: Invalid value"
    result = extract_relevant_stack_trace(stack_trace, message)
    assert result is not None
    assert result == stack_trace


def test_extract_relevant_stack_trace_long_with_error_type() -> None:
    """Test with a long stack trace and error type in message - should extract relevant part, i.e. end of the trace."""
    # Create a long stack trace
    stack_lines = [f"Line {i}: some code" for i in range(1, 16)]

    # Add the error at the end
    stack_lines.append("AssertionError: Expected 42 but got 24")

    stack_trace = "\n".join(stack_lines)
    message = "AssertionError: Expected 42 but got 24"

    result = extract_relevant_stack_trace(stack_trace, message)

    # Should include the error line
    assert result is not None
    assert "AssertionError: Expected 42 but got 24" in result
    # Should not include all lines (should be shorter than original)
    assert len(result.split("\n")) < len(stack_trace.split("\n"))


def test_extract_relevant_stack_trace_none() -> None:
    """Test with None stack trace."""
    result = extract_relevant_stack_trace(None, "Error message")
    assert result is None


def test_extract_relevant_stack_trace_with_complex_error_message() -> None:
    """Test with an error message containing multiple colons."""
    stack_trace = """
    File "/path/to/file.py", line 123, in some_function
    ValueError: Invalid URL: https://example.com/path
    """
    message = "ValueError: Invalid URL: https://example.com/path"
    result = extract_relevant_stack_trace(stack_trace, message)

    # Should find the error type correctly despite multiple colons
    assert result is not None
    assert "ValueError: Invalid URL: https://example.com/path" in result
