# py-videodev2

A Python script that automatically converts Video4Linux2 (V4L2) C header files to Python ctypes bindings.

## Overview

This tool parses the Linux `videodev2.h` header file and generates clean, Pythonic ctypes bindings for the V4L2 API. It handles complex C constructs like inline unions, enums, structs, and preprocessor definitions, producing ready-to-use Python code.

## Requirements

- Python 3.6+
- `clang` preprocessor (for header preprocessing)
- Linux kernel headers (for complete V4L2 definitions)

## Usage

### Basic Usage

```bash
./convert-videodev2.py /usr/include/linux/videodev2.h -o videodev2.py
```

### With Custom Defines

```bash
./convert-videodev2.py videodev2.h -D CUSTOM_DEFINE -D ANOTHER_DEFINE -o output.py
```

### With Custom Clang Binary

```bash
./convert-videodev2.py videodev2.h --bin /usr/bin/clang-15 -o output.py
```

### Command Line Options

- `input_file`: Path to the videodev2.h header file
- `-o, --output`: Output Python file (default: `videodev2.py`)
- `-D, --define`: Add preprocessor defines (default: `__aarch64__`, `__KERNEL__`, `__arch64__`)
- `--bin`: Path to clang binary (default: `clang`)

**Note**: The script includes `__aarch64__`, `__KERNEL__`, and `__arch64__` as default defines to ensure compatibility with ARM64 systems and enable kernel-specific definitions.

## Generated Code Structure

The generated Python module includes:

### Enums with Global Constants
```python
class v4l2_field(IntEnum):
    V4L2_FIELD_ANY = 0
    V4L2_FIELD_NONE = 1
    V4L2_FIELD_TOP = 2

# Global constants for compatibility
V4L2_FIELD_ANY = v4l2_field.V4L2_FIELD_ANY
V4L2_FIELD_NONE = v4l2_field.V4L2_FIELD_NONE
```

### Structures with Nested Unions
```python
class v4l2_plane(Structure):
    class _u(Union):
        _fields_ = [
            ('mem_offset', c_uint32),
            ('userptr', c_ulong),
            ('fd', c_int32)
        ]
    
    _fields_ = [
        ('bytesused', c_uint32),
        ('length', c_uint32),
        ('m', _u),
        ('data_offset', c_uint32),
        ('reserved', c_uint32 * 11)
    ]
```

### IOCTL Definitions
```python
VIDIOC_QUERYCAP = _IOR('V', 0, v4l2_capability)
VIDIOC_G_FMT = _IOWR('V', 4, v4l2_format)
```

### Helper Functions
```python
def v4l2_fourcc(a, b, c, d):
    """Create a FOURCC value from four characters"""
    return (ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24))
```

## Example Usage of Generated Code

```python
import videodev2 as v4l2
import fcntl

# Open video device
fd = open('/dev/video0', 'rb+', buffering=0)

# Query capabilities
cap = v4l2.v4l2_capability()
fcntl.ioctl(fd, v4l2.VIDIOC_QUERYCAP, cap)
print(f"Driver: {cap.driver.decode()}")
print(f"Card: {cap.card.decode()}")

# Set format
fmt = v4l2.v4l2_format()
fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
fmt.fmt.pix.width = 640
fmt.fmt.pix.height = 480
fmt.fmt.pix.pixelformat = v4l2.v4l2_fourcc('Y', 'U', 'Y', 'V')
fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)
```


## License

This project is licensed under the BSD 3-Clause License. See the script header for full license text.
