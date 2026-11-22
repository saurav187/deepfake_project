import logging

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from .errors import UNREADABLE, ApiError


def too_large(limit_mb: int):
    return ApiError(413, 'too_large', f'Choose a file that is {limit_mb} MB or smaller.')


async def read_upload(request: Request, limit_mb: int) -> tuple[bytes, str]:
    # Parse incrementally to avoid automatic spooling to disk and bound file and header bytes.
    limit = limit_mb * 1_000_000
    body_limit = limit + 64 * 1024
    try:
        length = int(request.headers.get('content-length', '0'))
    except ValueError:
        raise ApiError(422, 'unreadable', UNREADABLE) from None
    if length > body_limit:
        raise too_large(limit_mb)

    content_type, params = parse_options_header(request.headers.get('content-type', ''))
    boundary = params.get(b'boundary', b'')
    if content_type != b'multipart/form-data' or not 1 <= len(boundary) <= 70:
        raise ApiError(422, 'unreadable', UNREADABLE)

    data = bytearray()
    header_name = bytearray()
    header_value = bytearray()
    filename = None
    part_filename = None
    complete = False
    header_size = 0

    def header_chunk(target, chunk, start, end):
        nonlocal header_size
        header_size += end - start
        if header_size > 64 * 1024:
            raise ApiError(422, 'unreadable', UNREADABLE)
        target.extend(chunk[start:end])

    def header_end():
        nonlocal part_filename
        if header_name.lower() == b'content-disposition':
            kind, options = parse_options_header(bytes(header_value))
            if kind == b'form-data' and options.get(b'name') == b'file':
                part_filename = options.get(b'filename')
        header_name.clear()
        header_value.clear()

    def part_begin():
        nonlocal part_filename
        part_filename = None

    def headers_finished():
        nonlocal filename
        if part_filename:
            if filename is not None:
                raise ApiError(422, 'unreadable', UNREADABLE)
            filename = part_filename.decode('utf-8', errors='replace')

    def part_data(chunk, start, end):
        if part_filename:
            if len(data) + end - start > limit:
                raise too_large(limit_mb)
            data.extend(chunk[start:end])

    def finish():
        nonlocal complete
        complete = True

    parser = MultipartParser(boundary, {
        'on_part_begin': part_begin,
        'on_header_field': lambda *args: header_chunk(header_name, *args),
        'on_header_value': lambda *args: header_chunk(header_value, *args),
        'on_header_end': header_end,
        'on_headers_finished': headers_finished,
        'on_part_data': part_data,
        'on_end': finish,
    })
    # Parser diagnostics can include bytes from malformed input.
    parser.logger = logging.Logger("private_multipart", level=logging.CRITICAL + 1)  # noqa: LOG001
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > body_limit:
                raise too_large(limit_mb)
            if chunk:
                parser.write(chunk)
        parser.finalize()
    except (MultipartParseError, UnicodeError):
        raise ApiError(422, 'unreadable', UNREADABLE) from None

    if not complete or not filename:
        raise ApiError(422, 'unreadable', UNREADABLE)
    return bytes(data), filename
