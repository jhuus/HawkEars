#!/usr/bin/env python3


class ClassInfo:
    def __init__(
        self,
        name: str,
        code: str,
        alt_name: str,
        alt_code: str,
        index: int,
        include: bool,
    ):
        self.name = name
        self.code = code
        self.alt_name = alt_name
        self.alt_code = alt_code
        self.index = index
        self.include = include  # include in output?

    def __str__(self):
        return f"ClassInfo: name={self.name}, code={self.code}, include={self.include}"
