#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2024, Raspberry Pi Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Python script to convert videodev2.h to Python ctypes bindings
Usage: python convert-videodev2.py videodev2.h -o output_directory
"""

import re
import sys
import argparse
import datetime
import subprocess
import tempfile
import os
import shutil
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CDefine:
    name: str
    value: str
    is_function: bool = False
    is_problematic: bool = False


@dataclass
class CEnum:
    name: Optional[str]
    values: List[Tuple[str, Optional[str]]]  # (name, value)


@dataclass
class CField:
    type: str
    name: str
    array_size: Optional[str] = None


@dataclass
class CInlineUnion:
    """Represents an inline union within a struct"""

    field_name: str
    fields: List[CField]
    is_union: bool = True


@dataclass
class CStruct:
    name: str
    fields: List[CField]
    is_union: bool = False
    inline_unions: List[CInlineUnion] = None

    def __post_init__(self):
        if self.inline_unions is None:
            self.inline_unions = []


@dataclass
class CTypedef:
    old_type: str
    new_type: str


class VideoDevHeaderParser:
    def __init__(self):
        self.defines: List[CDefine] = []
        self.enums: List[CEnum] = []
        self.structs: List[CStruct] = []
        self.typedefs: List[CTypedef] = []

        # Type mappings from C to Python ctypes
        self.type_mapping = {
            "__u8": "c_uint8",
            "__u16": "c_uint16",
            "__u32": "c_uint32",
            "__u64": "c_uint64",
            "__s8": "c_int8",
            "__s16": "c_int16",
            "__s32": "c_int32",
            "__s64": "c_int64",
            "char": "c_char",
            "unsigned char": "c_uint8",
            "unsigned short": "c_uint16",
            "unsigned short int": "c_uint16",
            "unsigned int": "c_uint32",
            "unsigned long": "c_ulong",
            "unsigned long int": "c_ulong",
            "unsigned long long": "c_uint64",
            "signed char": "c_int8",
            "signed short": "c_int16",
            "signed short int": "c_int16",
            "signed int": "c_int",
            "signed long": "c_long",
            "signed long int": "c_long",
            "signed long long": "c_longlong",
            "int": "c_int",
            "long": "c_long",
            "long int": "c_long",
            "long long": "c_longlong",
            "void*": "c_void_p",
            "void": "c_void_p",
            "size_t": "c_size_t",
            "timeval": "c_longlong",
            "timespec": "c_longlong",
            "__kernel_timespec": "c_longlong",
            "__kernel_v4l2_timeval": "c_longlong",
            "__le32": "c_uint32",
            "__be32": "c_uint32",
            "__le64": "c_uint64",
            "__be64": "c_uint64",
        }

    def clean_line(self, line: str) -> str:
        """Remove extra whitespace (comments already removed by clang-format)"""
        return line.strip()

    def strip_user_qualifiers(self, content: str) -> str:
        """Strip __user qualifiers from all lines"""
        # Simply remove all occurrences of __user
        return content.replace("__user", "")

    def preprocess_with_cpp(
        self, content: str, defines: Optional[List[str]] = None, clang: str = "clang"
    ) -> str:
        """Use clang preprocessor to strip comments and resolve conditionals"""
        if defines is None:
            defines = []

        try:
            # Create a temporary file with .h extension
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".h", delete=False
            ) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Build the clang command with defines
            clang_cmd = [clang, "-E", "-P", "-dD"]
            if defines:
                for define in defines:
                    clang_cmd.extend(["-D", define])

            # Run clang preprocessor
            result = subprocess.run(
                clang_cmd + [temp_file_path], capture_output=True, text=True
            )

            # Display errors but continue with whatever output we got
            if result.returncode != 0:
                print(
                    f"Error running clang preprocessor: {result.stderr}",
                    file=sys.stderr,
                )

            preprocessed_content = result.stdout

            # Strip __user qualifiers from the preprocessed content
            preprocessed_content = self.strip_user_qualifiers(preprocessed_content)

            return preprocessed_content

        except Exception as e:
            print(f"Error in preprocess_with_cpp: {e}", file=sys.stderr)
            return content
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def remove_multiline_comments_regex(self, content: str) -> str:
        """Improved regex-based comment removal"""
        result = []
        i = 0
        while i < len(content):
            # Look for start of comment
            if i < len(content) - 1 and content[i : i + 2] == "/*":
                # Find end of comment
                end = content.find("*/", i + 2)
                if end != -1:
                    # Replace comment with equivalent whitespace to preserve structure
                    comment_text = content[i : end + 2]
                    # Count newlines and preserve them
                    newlines = comment_text.count("\n")
                    result.append("\n" * newlines)
                    i = end + 2
                else:
                    # Unclosed comment, just add the character
                    result.append(content[i])
                    i += 1
            elif i < len(content) - 1 and content[i : i + 2] == "//":
                # Find end of line
                end = content.find("\n", i)
                if end != -1:
                    result.append("\n")  # Preserve the newline
                    i = end + 1
                else:
                    # Comment goes to end of file
                    break
            else:
                result.append(content[i])
                i += 1

        return "".join(result)

    def parse_define(self, line: str) -> Optional[CDefine]:
        """Parse #define statements, including function-like macros"""

        # First try to match function-like macros
        func_match = re.match(r"#define\s+(\w+)\(([^)]*)\)\s+(.+)", line)
        if func_match:
            name = func_match.group(1)
            params = func_match.group(2).strip()
            body = func_match.group(3).strip()

            # Only process V4L2-related function-like macros that are self-contained
            if name.startswith('v4l2_'):
                # Clean up the function body for Python
                python_body = self.convert_c_expression_to_python(body)

                # Create Python function definition
                python_func = f"def {name}({params}):\n    return {python_body}"
                return CDefine(name, python_func, is_function=True)
            elif name.startswith('_IO'):
                # Handle IOCTL macros (now that _IOC_ constants are included)
                python_body = self.convert_c_expression_to_python(body)

                # Handle parameter name conflicts (e.g., 'type' is reserved in Python)
                if 'type' in params:
                    params = params.replace('type', 'ioc_type')
                    python_body = python_body.replace('type', 'ioc_type')

                # For IOCTL functions that pass ioc_type to _IOC, add ord() conversion
                if name in ['_IO', '_IOR', '_IOW', '_IOWR'] and 'ioc_type' in python_body:
                    python_body = python_body.replace('(ioc_type)', 'ord(ioc_type)')

                # For _IOC function, wrap result in c_int32 to get proper signed integer
                if name == '_IOC':
                    python_body = f"ctypes.c_int32({python_body}).value"

                # Create Python function definition
                python_func = f"def {name}({params}):\n    return {python_body}"
                return CDefine(name, python_func, is_function=True)

            return None

        # Handle regular #define statements
        match = re.match(r"#define\s+(\w+)\s+(.+)", line)
        if match:
            name = match.group(1)
            value = match.group(2).strip()

            # Skip compiler-specific defines and problematic ones
            if (name.startswith('__') and name.endswith('__')) or \
               name in ['__GNUC__', '__STDC__', '__STDC_VERSION__'] or \
               'ULL' in value or 'UL' in value or value.endswith('L') or \
               'GENMASK' in value or 'BIT' in value:  # Skip macros that use undefined kernel functions
                return None

            # Mark problematic constants that cause dependency issues
            is_problematic = name in ['V4L2_PIX_FMT_HM12', 'V4L2_PIX_FMT_SUNXI_TILED_NV12']

            # Only include V4L2-related defines, VIDIOC defines, and IOCTL-related constants
            if not (name.startswith('V4L2_') or name.startswith('VIDIOC_') or name.startswith('BASE_') or name.startswith('_IOC_')):
                return None

            # Remove inline comments
            value = re.sub(r"/\*.*?\*/", "", value)
            value = re.sub(r"//.*", "", value)
            value = value.strip()

            # Remove U, UL, ULL suffixes from integer literals in constants too
            value = re.sub(r'\b(\d+)ULL\b', r'\1', value)
            value = re.sub(r'\b(\d+)UL\b', r'\1', value)
            value = re.sub(r'\b(\d+)U\b', r'\1', value)

            # Handle mathematical expressions in constants (like _IOC_TYPESHIFT = (_IOC_NRSHIFT+_IOC_NRBITS))
            # These are valid Python expressions once the referenced constants are defined

            # Handle C-style casts like ((type)value) -> value
            cast_match = re.match(r"\(\(([^)]+)\)\s*([^)]+)\)", value)
            if cast_match:
                # Extract just the value part, removing the cast
                value = cast_match.group(2).strip()

            # Handle struct references in IOCTL definitions like "struct v4l2_capability" -> "v4l2_capability"
            value = re.sub(r"\bstruct\s+(\w+)", r"\1", value)

            # Convert basic C types to ctypes in IOCTL definitions
            value = re.sub(r'\bint\b', 'c_int', value)
            value = re.sub(r'\bunsigned int\b', 'c_uint', value)
            value = re.sub(r'\blong\b', 'c_long', value)
            value = re.sub(r'\bunsigned long\b', 'c_ulong', value)
            value = re.sub(r'\bshort\b', 'c_short', value)
            value = re.sub(r'\bunsigned short\b', 'c_ushort', value)
            value = re.sub(r'\bchar\b', 'c_char', value)
            value = re.sub(r'\bunsigned char\b', 'c_uchar', value)

            # For VIDIOC defines, check if they reference undefined structs
            if name.startswith('VIDIOC_'):
                # Get list of defined struct names for validation
                defined_structs = {struct.name for struct in self.structs}

                # Extract struct names that look like v4l2_* from IOCTL definition
                struct_refs = re.findall(r'\bv4l2_\w+', value)
                undefined_structs = [s for s in struct_refs if s not in defined_structs]

                if undefined_structs:
                    print(f"⚠️  Skipping {name} - references undefined struct(s): {', '.join(undefined_structs)}")
                    return None

            return CDefine(name, value, is_function=False, is_problematic=is_problematic)
        return None

    def convert_c_expression_to_python(self, c_expr: str) -> str:
        """Convert C expression to Python equivalent"""
        python_expr = c_expr

        # Remove C type casts like (__u32), (__u16), etc.
        python_expr = re.sub(r'\(__u\d+\)', '', python_expr)
        python_expr = re.sub(r'\(__s\d+\)', '', python_expr)
        python_expr = re.sub(r'\(unsigned\s+\w+\)', '', python_expr)
        python_expr = re.sub(r'\(signed\s+\w+\)', '', python_expr)
        python_expr = re.sub(r'\(int\)', '', python_expr)

        # Remove U, UL, ULL suffixes from integer literals
        python_expr = re.sub(r'\b(\d+)ULL\b', r'\1', python_expr)
        python_expr = re.sub(r'\b(\d+)UL\b', r'\1', python_expr)
        python_expr = re.sub(r'\b(\d+)U\b', r'\1', python_expr)

        # Handle sizeof expressions and fix typos
        python_expr = re.sub(r'\bsizeof\s*\(\s*struct\s+(\w+)\s*\)', r'ctypes.sizeof(\1)', python_expr)
        python_expr = re.sub(r'\bsizeof\s*\(\s*(\w+)\s*\)', r'ctypes.sizeof(\1)', python_expr)
        python_expr = re.sub(r'\bcioc_types\.sizeof\b', 'ctypes.sizeof', python_expr)

        # For v4l2_fourcc, convert bare parameter references to ord() calls
        # This handles cases like: (a) | (b) << 8 | (c) << 16 | (d) << 24
        if any(param in python_expr for param in ['(a)', '(b)', '(c)', '(d)']):
            python_expr = re.sub(r'\(([abcd])\)', r'ord(\1)', python_expr)

        # Clean up extra spaces and parentheses from removed casts
        python_expr = re.sub(r'\(\s*\)', '', python_expr)  # Remove empty parentheses
        python_expr = re.sub(r'\s+', ' ', python_expr)
        python_expr = python_expr.strip()

        return python_expr

    def parse_enum(self, content: str, start_pos: int) -> Tuple[Optional[CEnum], int]:
        """Parse enum definitions"""
        lines = content[start_pos:].split("\n")
        enum_match = re.match(r"enum\s+(\w+)?\s*{", lines[0])
        if not enum_match:
            return None, 0

        enum_name = enum_match.group(1)
        values = []

        i = 1
        brace_count = 1
        current_value = 0

        while i < len(lines) and brace_count > 0:
            line = self.clean_line(lines[i])
            if not line:
                i += 1
                continue

            brace_count += line.count("{") - line.count("}")

            if brace_count == 0:
                break

            # Parse enum values
            for item in line.split(","):
                item = item.strip().rstrip(",")
                if not item or item in ["{", "}"]:
                    continue

                if "=" in item:
                    name, val = item.split("=", 1)
                    name = name.strip()
                    val = val.strip()
                    # Basic validation - must start with letter or underscore
                    if name and (name[0].isalpha() or name[0] == "_"):
                        values.append((name, val))
                        try:
                            current_value = int(val, 0) + 1
                        except ValueError:
                            current_value += 1
                else:
                    # Basic validation - must start with letter or underscore
                    if item and (item[0].isalpha() or item[0] == "_"):
                        values.append((item, str(current_value)))
                        current_value += 1
            i += 1

        return CEnum(enum_name, values), i

    def parse_struct(
        self, content: str, start_pos: int
    ) -> Tuple[Optional[CStruct], int]:
        """Parse struct/union definitions"""
        lines = content[start_pos:].split("\n")
        struct_match = re.match(r"(struct|union)\s+(\w+)?\s*{", lines[0])
        if not struct_match:
            return None, 0

        is_union = struct_match.group(1) == "union"
        struct_name = struct_match.group(2)
        fields = []
        inline_unions = []

        # Set current struct name for self-reference detection
        self._current_struct_name = struct_name

        i = 1
        brace_count = 1

        while i < len(lines) and brace_count > 0:
            line = self.clean_line(lines[i])
            if not line:
                i += 1
                continue

            # Check for inline union/struct BEFORE updating brace count
            if brace_count == 1 and (
                line.strip() == "union {"
                or line.strip() == "struct {"
                or "union {" in line
                or "struct {" in line
            ):
                # Parse inline union/struct
                remaining_content = "\n".join(lines[i:])
                inline_union, lines_processed = self.parse_inline_union_struct(
                    remaining_content, 0, struct_name
                )
                if inline_union:
                    inline_unions.append(inline_union)
                    # Create a field that references the nested class name
                    # Use _u for unions, _s for structs (add number if multiple of same type)
                    base_name = "_u" if inline_union.is_union else "_s"
                    # Check if we already have a union/struct of this type in this struct
                    existing_count = sum(
                        1
                        for iu in inline_unions[:-1]
                        if iu.is_union == inline_union.is_union
                    )
                    if existing_count > 0:
                        nested_class_name = f"{base_name}{existing_count + 1}"
                    else:
                        nested_class_name = base_name
                    fields.append(CField(nested_class_name, inline_union.field_name))
                    # Skip all the lines that were processed by parse_inline_union_struct
                    i += lines_processed
                    continue  # Important: continue to avoid processing the same lines again
                else:
                    i += 1
                    continue

            brace_count += line.count("{") - line.count("}")

            if brace_count == 0:
                break

            # Parse regular field declarations
            if brace_count == 1 and line.endswith(";"):
                field = self.parse_field(line.rstrip(";"))
                if field:
                    # Check for self-references and convert to pointer
                    if (
                        hasattr(self, "_current_struct_name")
                        and field.type == self._current_struct_name
                    ):
                        field.type = "POINTER(" + field.type + ")"
                    fields.append(field)

            i += 1

        if struct_name:
            struct = CStruct(struct_name, fields, is_union)
            struct.inline_unions = inline_unions
            # Clear current struct name
            self._current_struct_name = None
            return struct, i

        # Clear current struct name even if struct_name is None
        self._current_struct_name = None
        return None, i

    def parse_inline_union_struct(
        self, content: str, start_pos: int, parent_struct_name: str = None
    ) -> Tuple[Optional[CInlineUnion], int]:
        """Parse inline union/struct within a struct and return information for nested class generation"""
        lines = content[start_pos:].split("\n")

        # Match inline union/struct pattern like "union { ... } field_name;"
        first_line = self.clean_line(lines[0])
        if not ("union {" in first_line or "struct {" in first_line):
            return None, 0

        is_union = "union {" in first_line

        # Find the closing brace and field name
        i = 0
        brace_count = 0
        union_fields = []

        for line in lines:
            clean_line = self.clean_line(line)
            if not clean_line:
                i += 1
                continue

            brace_count += clean_line.count("{") - clean_line.count("}")

            # Parse fields inside the union/struct
            if (
                brace_count == 1
                and clean_line.endswith(";")
                and clean_line != first_line
            ):
                field = self.parse_field(clean_line.rstrip(";"))
                if field:
                    union_fields.append(field)

            # Check if we've reached the end with field name
            if brace_count == 0 and i > 0:
                # Extract field name after closing brace
                # Pattern: "} field_name;"
                field_name_match = re.search(r"}\s*(\w+)\s*;", clean_line)
                if field_name_match:
                    field_name = field_name_match.group(1)

                    # Return inline union information for nested class generation
                    return CInlineUnion(field_name, union_fields, is_union), i + 1
                break

            i += 1

        return None, i

    def parse_field(self, line: str) -> Optional[CField]:
        """Parse struct field declarations"""
        original_line = line
        line = line.strip()

        # Skip empty lines
        if not line:
            return None

        # Skip lines that are just braces or comments
        if line in ["{", "}"] or line.startswith("//") or line.startswith("/*"):
            return None

        # Handle struct type declarations like "struct v4l2_fract min" or "struct v4l2_clip __user next"
        struct_match = re.match(r"struct\s+(\w+)(?:\s+__\w+)?\s+(\w+)", line)
        if struct_match:
            field_type = struct_match.group(
                1
            )  # Extract the struct name without 'struct'
            field_name = struct_match.group(2)
            return CField(field_type, field_name)

        # Handle union type declarations like "union v4l2_fract min" or "union v4l2_something __user field"
        union_match = re.match(r"union\s+(\w+)(?:\s+__\w+)?\s+(\w+)", line)
        if union_match:
            field_type = union_match.group(1)  # Extract the union name without 'union'
            field_name = union_match.group(2)
            return CField(field_type, field_name)

        # Handle array declarations like "char name[32]"
        array_match = re.match(r"(.+?)\s+(\w+)\[([^\]]+)\]", line)
        if array_match:
            field_type = array_match.group(1).strip()
            field_name = array_match.group(2)
            array_size = array_match.group(3)

            # Validate field type is not empty
            if not field_type:
                return None

            # Evaluate mathematical expressions in array size
            try:
                # Simple evaluation for common mathematical expressions
                # Replace common constants that might not be defined yet
                array_size_eval = array_size.strip()
                # Handle simple arithmetic expressions
                if re.match(r"^[\d\s+\-*/()]+$", array_size_eval):
                    array_size = str(eval(array_size_eval))
                # If it contains identifiers, leave it as-is (it might be a constant)
            except:
                # If evaluation fails, use the original array size
                pass

            return CField(field_type, field_name, array_size)

        # Handle pointer declarations like "struct v4l2_plane *planes" or "void *base" or "struct v4l2_clip __user *next"
        pointer_match = re.match(r"(.+?)\s+\*(\w+)", line)
        if pointer_match:
            field_type = pointer_match.group(1).strip() + "*"  # Add * to type
            field_name = pointer_match.group(2)

            # Clean up type - remove qualifiers like __user, __kernel, etc.
            field_type = re.sub(r"\b__\w+\b", "", field_type).strip()
            # Remove extra spaces
            field_type = re.sub(r"\s+", " ", field_type).strip()

            # Validate field name
            if not field_name or not field_name.isidentifier():
                print(
                    f"Warning: Invalid field name '{field_name}' in line: '{original_line}'",
                    file=sys.stderr,
                )
                return None

            return CField(field_type, field_name)

        # Handle regular field declarations like "int width"
        regular_match = re.match(r"(.+?)\s+(\w+)", line)
        if regular_match:
            field_type = regular_match.group(1).strip()
            field_name = regular_match.group(2)

            # Validate field type is not empty
            if not field_type:
                return None

            # Validate field name
            if not field_name or not field_name.isidentifier():
                print(
                    f"Warning: Invalid field name '{field_name}' in line: '{original_line}'",
                    file=sys.stderr,
                )
                return None

            return CField(field_type, field_name)

        return None

    def parse_typedef(self, line: str) -> Optional[CTypedef]:
        """Parse typedef statements"""
        match = re.match(r"typedef\s+(.+?)\s+(\w+);", line)
        if match:
            old_type = match.group(1).strip()
            new_type = match.group(2)
            return CTypedef(old_type, new_type)
        return None

    def parse_header(
        self, content: str, defines: Optional[List[str]] = None, clang: str = "clang"
    ):
        """Parse the entire header file"""
        # First preprocess with cc to remove comments
        content = self.preprocess_with_cpp(content, defines, clang)
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = self.clean_line(lines[i])

            if not line:
                i += 1
                continue

            # Parse #define
            if line.startswith("#define"):
                define = self.parse_define(line)
                if define:
                    self.defines.append(define)

            # Parse typedef
            elif line.startswith("typedef"):
                typedef = self.parse_typedef(line)
                if typedef:
                    self.typedefs.append(typedef)

            # Parse enum
            elif "enum" in line and "{" in line:
                remaining_content = "\n".join(lines[i:])
                enum, lines_processed = self.parse_enum(remaining_content, 0)
                if enum:
                    self.enums.append(enum)
                    # Skip processed lines
                    i += lines_processed
                else:
                    i += 1

            # Parse struct/union
            elif ("struct" in line or "union" in line) and "{" in line:
                remaining_content = "\n".join(lines[i:])
                struct, lines_processed = self.parse_struct(remaining_content, 0)
                if struct:
                    self.structs.append(struct)
                    # Skip processed lines
                    i += lines_processed
                else:
                    i += 1

            i += 1

    def convert_type(self, c_type: str) -> str:
        """Convert C type to Python ctypes type"""
        original_type = c_type
        c_type = c_type.strip()

        # Handle empty types - this shouldn't happen, so let's debug it
        if not c_type:
            print(
                f"Warning: Empty type encountered in convert_type, original: '{original_type}'",
                file=sys.stderr,
            )
            return "c_void_p"

        # Strip kernel annotations like __user, __kernel, __le32, etc.
        # But preserve basic kernel types and actual type names like __m_union, __fmt_struct
        basic_kernel_types = {
            "__u8",
            "__u16",
            "__u32",
            "__u64",
            "__s8",
            "__s16",
            "__s32",
            "__s64",
            "__le32",
            "__be32",
        }
        actual_type_patterns = {
            "__m_union",
            "__fmt_struct",
            "__raw_struct",
            "__stop_struct",
            "__start_struct",
            "__fmt_union",
            "__parm_union",
            "__u_union",
        }
        words = c_type.split()
        filtered_words = []
        for word in words:
            # Keep basic kernel types, actual type names, and non-kernel annotations
            if (
                not word.startswith("__")
                or word in basic_kernel_types
                or word in actual_type_patterns
                or any(pattern in word for pattern in ["_union", "_struct"])
            ):
                filtered_words.append(word)
        c_type = " ".join(filtered_words).strip()

        # Handle struct references like "struct v4l2_capability" -> "v4l2_capability"
        c_type = re.sub(r"\bstruct\s+(\w+)", r"\1", c_type)

        # Handle union references like "union v4l2_capability" -> "v4l2_capability"
        c_type = re.sub(r"\bunion\s+(\w+)", r"\1", c_type)

        # Handle enum references like "enum v4l2_av1_warp_model" -> "v4l2_av1_warp_model"
        c_type = re.sub(r"\benum\s+(\w+)", r"\1", c_type)

        # Clean up extra whitespace
        c_type = re.sub(r"\s+", " ", c_type).strip()

        # Check again after cleaning
        if not c_type:
            print(
                f"Warning: Type became empty after cleaning, original: '{original_type}'",
                file=sys.stderr,
            )
            return "c_void_p"

        # Handle pointer types (including complex ones like "type *")
        if "*" in c_type:
            # Extract the base type by removing * and whitespace
            base_type = c_type.replace("*", "").strip()
            if not base_type or base_type == "void":
                return "c_void_p"
            elif base_type == "char":
                return "c_char_p"
            else:
                converted_base = self.convert_type(base_type)
                return f"POINTER({converted_base})"

        # Handle const types
        if c_type.startswith("const "):
            return self.convert_type(c_type[6:])

        # Handle special void type
        if c_type == "void":
            return "c_void_p"

        # Handle bare 'enum' type (when specific enum name is missing)
        if c_type == "enum":
            return "c_uint32"

        # Direct mapping
        if c_type in self.type_mapping:
            return self.type_mapping[c_type]

        # Check if it's a custom type we've defined
        for typedef in self.typedefs:
            if typedef.new_type == c_type:
                return self.convert_type(typedef.old_type)

        # Check if it's a struct/union we've defined
        for struct in self.structs:
            if struct.name == c_type:
                return c_type

        # Handle self-references - if a struct tries to reference itself, it should be a pointer
        # This is a workaround for cases where pointer parsing missed the asterisk
        if (
            hasattr(self, "_current_struct_name")
            and self._current_struct_name == c_type
        ):
            return "POINTER(" + c_type + ")"

        # Check if it's an enum we've defined - convert to underlying integer type
        for enum in self.enums:
            if enum.name == c_type:
                return "c_uint32"  # Enums are typically 32-bit integers

        # Handle basic C types that might not be in mapping
        if c_type in ["unsigned", "signed"]:
            return "c_uint"

        # For unknown types, assume they're valid class names but warn about potential issues
        if c_type and c_type.replace("_", "").replace(" ", "").isalnum():
            return c_type

        # Fallback for problematic types - provide more specific error info
        if not c_type:
            print(
                f"Warning: Empty type string encountered, original: '{original_type}'",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: Unknown type '{c_type}', using c_void_p, original: '{original_type}'",
                file=sys.stderr,
            )
        return "c_void_p"



    def generate_python_code(self) -> str:
        """Generate Python ctypes code"""
        output = []

        # Header
        current_year = datetime.datetime.now().year
        output.append('# SPDX-License-Identifier: BSD-3-Clause')
        output.append('#')
        output.append(f'# Copyright (C) {current_year}, Raspberry Pi Ltd.')
        output.append('#')
        output.append('# Redistribution and use in source and binary forms, with or without')
        output.append('# modification, are permitted provided that the following conditions are met:')
        output.append('#')
        output.append('# 1. Redistributions of source code must retain the above copyright notice, this')
        output.append('#    list of conditions and the following disclaimer.')
        output.append('#')
        output.append('# 2. Redistributions in binary form must reproduce the above copyright notice,')
        output.append('#    this list of conditions and the following disclaimer in the documentation')
        output.append('#    and/or other materials provided with the distribution.')
        output.append('#')
        output.append('# 3. Neither the name of the copyright holder nor the names of its')
        output.append('#    contributors may be used to endorse or promote products derived from')
        output.append('#    this software without specific prior written permission.')
        output.append('#')
        output.append('# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"')
        output.append('# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE')
        output.append('# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE')
        output.append('# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE')
        output.append('# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL')
        output.append('# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR')
        output.append('# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER')
        output.append('# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,')
        output.append('# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE')
        output.append('# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.')
        output.append('')
        output.append('"""')
        output.append('Python ctypes bindings for Video4Linux2 (V4L2) API')
        output.append('Auto-generated from videodev2.h')
        output.append('"""')
        output.append('')
        output.append('import ctypes')
        output.append('from ctypes import (')
        output.append('    Structure, Union, Array, POINTER,')
        output.append('    c_uint8, c_uint16, c_uint32, c_uint64,')
        output.append('    c_int8, c_int16, c_int32, c_int64,')
        output.append('    c_char, c_void_p, c_long, c_ulong, c_char_p, c_size_t, c_int, c_uint, c_longlong')
        output.append(')')
        output.append('from enum import IntEnum')
        output.append('')

        # Helper function for converting c_uint8 arrays to strings
        output.append('# Helper function for converting c_uint8 arrays to strings')
        output.append('def arr_to_str(arr, encoding="utf-8", errors="ignore", null_terminated=True):')
        output.append('    """')
        output.append('    Convert a ctypes c_uint8 array to a Python string.')
        output.append('    ')
        output.append('    Args:')
        output.append('        arr: ctypes c_uint8 array')
        output.append('        encoding: Text encoding (default: utf-8)')
        output.append('        errors: How to handle decode errors (ignore/replace/strict)')
        output.append('        null_terminated: Whether to stop at first null byte (common in C strings)')
        output.append('    ')
        output.append('    Returns:')
        output.append('        Python string')
        output.append('    """')
        output.append('    byte_data = bytes(arr)')
        output.append('    if null_terminated:')
        output.append('        null_pos = byte_data.find(b"\\x00")')
        output.append('        if null_pos != -1:')
        output.append('            byte_data = byte_data[:null_pos]')
        output.append('    return byte_data.decode(encoding, errors=errors)')
        output.append('')

        # Build __all__ list for explicit exports (especially for functions starting with _)
        all_exports = []

        # Add the helper function
        all_exports.append("'arr_to_str'")



        # Add function names from macros
        for define in self.defines:
            if define.is_function:
                all_exports.append(f"'{define.name}'")

        # Add enum names and values
        for enum in self.enums:
            if enum.name:
                all_exports.append(f"'{enum.name}'")
                for name, _ in enum.values:
                    all_exports.append(f"'{name}'")

        # Add struct names
        for struct in self.structs:
            all_exports.append(f"'{struct.name}'")

        # Add typedef names
        for typedef in self.typedefs:
            all_exports.append(f"'{typedef.new_type}'")

        # Add constant names (excluding problematic ones)
        for define in self.defines:
            if not define.is_function and not define.is_problematic:
                all_exports.append(f"'{define.name}'")

        if all_exports:
            output.append('# Explicit exports for star imports')
            output.append('__all__ = [')
            # Group exports by type for readability
            output.append('    # All generated symbols')
            for i, export in enumerate(all_exports):
                if i == len(all_exports) - 1:
                    output.append(f'    {export}')
                else:
                    output.append(f'    {export},')
            output.append(']')
            output.append('')

        # Type aliases
        output.append("# Type aliases for kernel types")
        type_aliases = [
            ("__u8", "c_uint8"),
            ("__u16", "c_uint16"),
            ("__u32", "c_uint32"),
            ("__u64", "c_uint64"),
            ("__s8", "c_int8"),
            ("__s16", "c_int16"),
            ("__s32", "c_int32"),
            ("__s64", "c_int64"),
        ]

        for c_type, py_type in type_aliases:
            output.append(f"{c_type} = {py_type}")
        output.append("")

        # Typedefs
        if self.typedefs:
            output.append("# Typedefs")
            for typedef in self.typedefs:
                if typedef.new_type not in [alias[0] for alias in type_aliases]:
                    converted_type = self.convert_type(typedef.old_type)
                    output.append(f"{typedef.new_type} = {converted_type}")
            output.append("")

        # Enums
        for enum in self.enums:
            if enum.name:
                output.append(f"class {enum.name}(IntEnum):")
                if enum.values:
                    for name, value in enum.values:
                        if isinstance(value, str) and value.startswith("V4L2_"):
                            # This is a reference to another enum value
                            output.append(f"    {name} = {value}")
                        else:
                            output.append(f"    {name} = {value}")
                else:
                    output.append("    pass")
                output.append("")

                # Export enum values as global constants immediately after enum definition
                for name, value in enum.values:
                    output.append(f"{name} = {enum.name}.{name}")
                output.append("")

        # Structures - define these before constants that reference them
        for struct in self.structs:
            if struct.is_union:
                output.append(f"class {struct.name}(Union):")
            else:
                output.append(f"class {struct.name}(Structure):")

            # Generate nested classes for inline unions first
            if struct.inline_unions:
                union_count = 0
                struct_count = 0
                for inline_union in struct.inline_unions:
                    # Use same naming convention as in parsing
                    if inline_union.is_union:
                        union_count += 1
                        nested_class_name = (
                            "_u" if union_count == 1 else f"_u{union_count}"
                        )
                    else:
                        struct_count += 1
                        nested_class_name = (
                            "_s" if struct_count == 1 else f"_s{struct_count}"
                        )

                    union_type = "Union" if inline_union.is_union else "Structure"
                    output.append(f"    class {nested_class_name}({union_type}):")
                    output.append("        _fields_ = [")
                    for field in inline_union.fields:
                        py_type = self.convert_type(field.type)
                        if field.array_size:
                            py_type = f"{py_type} * {field.array_size}"
                        output.append(f"            ('{field.name}', {py_type}),")
                    output.append("        ]")
                    output.append("")

            # Check if this struct has self-references
            has_self_reference = False
            if struct.fields:
                for field in struct.fields:
                    py_type = self.convert_type(field.type)
                    if f"POINTER({struct.name})" in py_type:
                        has_self_reference = True
                        break

            # Generate main struct fields
            if has_self_reference:
                # Use forward declaration pattern for self-referencing structs
                output.append("    pass")
                output.append(f"{struct.name}._fields_ = [")
                for field in struct.fields:
                    py_type = self.convert_type(field.type)
                    if field.array_size:
                        py_type = f"{py_type} * {field.array_size}"
                    output.append(f"    ('{field.name}', {py_type}),")
                output.append("]")
            else:
                # Normal struct definition
                if struct.fields:
                    output.append("    _fields_ = [")
                    for field in struct.fields:
                        py_type = self.convert_type(field.type)
                        if field.array_size:
                            py_type = f"{py_type} * {field.array_size}"
                        output.append(f"        ('{field.name}', {py_type}),")
                    output.append("    ]")
                else:
                    output.append("    pass")
            output.append("")

        # Import fcntl for IOCTL operations
        output.append("import fcntl")
        output.append("")

        # Functions and constants from #define - these come LAST so they can reference enum exports
        if self.defines:
            # Separate constants that depend on functions vs those that don't
            functions = [d for d in self.defines if d.is_function]
            constants = [d for d in self.defines if not d.is_function]

            # Split constants into those that use functions and those that don't
            basic_constants = []
            function_dependent_constants = []

            for const in constants:
                if any(func.name in const.value for func in functions):
                    function_dependent_constants.append(const)
                else:
                    basic_constants.append(const)

            # Output order: basic constants, then functions, then function-dependent constants
            if basic_constants:
                output.append("# Basic constants")
                for define in basic_constants:
                    if define.is_problematic:
                        output.append(f"# {define.name} = {define.value}  # Commented out due to dependency issues")
                    else:
                        output.append(f"{define.name} = {define.value}")

                # Add missing computed IOCTL constants that may not be captured properly
                output.append("# Computed IOCTL constants")
                output.append("_IOC_TYPESHIFT = 8")
                output.append("_IOC_SIZESHIFT = 16")
                output.append("_IOC_DIRSHIFT = 30")
                output.append("")

            if functions:
                output.append("# Helper functions from macros")
                for define in functions:
                    output.append(define.value)
                    output.append("")

            if function_dependent_constants:
                output.append("# Constants that use helper functions")
                for define in function_dependent_constants:
                    if define.is_problematic:
                        output.append(f"# {define.name} = {define.value}  # Commented out due to dependency issues")
                    else:
                        output.append(f"{define.name} = {define.value}")
                output.append("")



        return "\n".join(output)

    def test_import(self, output_file: str) -> bool:
        """Test importing the generated Python file to validate syntax"""
        if not output_file or output_file == "-":
            print("Cannot test import: output written to stdout", file=sys.stderr)
            return True

        # Get the module name from the file path
        module_name = os.path.splitext(os.path.basename(output_file))[0]

        try:
            # Save current working directory
            original_cwd = os.getcwd()

            # Change to the directory containing the output file
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if output_dir:
                os.chdir(output_dir)

            # Try to import the module
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                module_name, os.path.abspath(output_file)
            )
            if spec is None:
                print(
                    f"Error: Could not create module spec for {output_file}",
                    file=sys.stderr,
                )
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return True

        except SyntaxError as e:
            print(f"✗ Syntax error in generated file {output_file}:")
            print(f"  Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
            print(f"  {e.msg}")
            return False
        except ImportError as e:
            print(f"✗ Import error in generated file {output_file}: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error importing {output_file}: {e}")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)


def create_package_directory(output_dir: str, package_name: str = "videodev2") -> str:
    """Create package directory structure for PyPI publishing"""
    # Create package directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, package_name))

    return output_dir


def copy_template_files(package_dir: str, package_name: str = "videodev2") -> None:
    """Copy template files to the package directory"""
    # Copy setup.py.template to setup.py
    if os.path.exists("setup.py.template"):
        shutil.copy2("setup.py.template", os.path.join(package_dir, "setup.py"))
    else:
        print("⚠️  Warning: setup.py.template not found")

    # Create README.md from template
    create_package_readme(package_dir, package_name)


def create_package_readme(package_dir: str, package_name: str = "videodev2") -> None:
    """Create a README.md file for the package from template."""
    readme_file = os.path.join(package_dir, "README.md")
    try:
        # Read the README.md.template
        if os.path.exists("README.md.template"):
            with open("README.md.template", "r") as f:
                template_content = f.read()

            # Replace the placeholder with the package name
            readme_content = template_content.replace("{{PACKAGE_NAME}}", package_name)

            with open(readme_file, "w") as f:
                f.write(readme_content)
            print(f"✅ Successfully generated {readme_file}")
        else:
            print(f"❌ README.md.template not found", file=sys.stderr)
    except IOError as e:
        print(f"❌ Error writing README.md: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Convert videodev2.h to Python ctypes bindings')
    parser.add_argument('input_file', help='Input videodev2.h file')
    parser.add_argument('-o', '--output', help='Output directory (default: ./videodev2-package)', default='./videodev2-package')
    parser.add_argument('-D', '--define', action='append', help='Add preprocessor defines', default=['__aarch64__', '__KERNEL__', '__arch64__'])
    parser.add_argument('--bin', help='Path to clang binary (default: clang)', default='clang')
    parser.add_argument('--publish', action='store_true', help='Publish generated package to PyPI after successful generation')
    parser.add_argument('--package-name', help='Package name for PyPI (default: videodev2)', default='videodev2')
    parser.add_argument('--test-pypi', action='store_true', help='Publish to TestPyPI instead of PyPI')

    args = parser.parse_args()

    # Read input file
    try:
        with open(args.input_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        return 1
    except IOError as e:
        print(f"❌ Error reading input file: {e}", file=sys.stderr)
        return 1

    # Parse header
    parser = VideoDevHeaderParser()
    parser.parse_header(content, args.define, args.bin)

    # Generate Python code
    python_code = parser.generate_python_code()

    # Create package directory structure
    print(f"📦 Creating package directory: {args.output}")
    package_dir = create_package_directory(args.output, args.package_name)

    # Copy template files
    print(f"📋 Copying template files...")
    copy_template_files(package_dir, args.package_name)

    # Create the generated V4L2 bindings file
    videodev2_file = os.path.join(package_dir, args.package_name, "videodev2.py")
    try:
        with open(videodev2_file, "w") as f:
            f.write(python_code)
        print(f"✅ Successfully generated {videodev2_file}")
    except IOError as e:
        print(f"❌ Error writing videodev2.py file: {e}", file=sys.stderr)
        return 1

    # Create the __init__.py file from template
    package_file = os.path.join(package_dir, args.package_name, "__init__.py")
    try:
        # Read the __init__.py.template
        if os.path.exists("__init__.py.template"):
            with open("__init__.py.template", "r") as f:
                template_content = f.read()

            with open(package_file, "w") as f:
                f.write(template_content)
            print(f"✅ Successfully generated {package_file}")
        else:
            print(f"❌ __init__.py.template not found", file=sys.stderr)
            return 1
    except IOError as e:
        print(f"❌ Error writing __init__.py file: {e}", file=sys.stderr)
        return 1

    # Test import
    if parser.test_import(os.path.abspath(videodev2_file)):
        print(f"✅ Import test successful: {videodev2_file} imports without errors")
        print(f"📁 Package ready for manual publishing:")
        print(f"   cd {args.output}")
        print(f"   python -m build")
        print(f"   twine upload dist/*")

    return 0


if __name__ == "__main__":
    sys.exit(main())
