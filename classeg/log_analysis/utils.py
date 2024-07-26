from typing import TextIO


def clean_line(line):
    return ':'.join(line.split(':')[2:])


def readline(log_file: TextIO):
    return clean_line(log_file.readline()).strip()
