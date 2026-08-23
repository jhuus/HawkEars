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
        *,
        model_name: str | None = None,
        model_code: str | None = None,
    ):
        self.name = name
        self.code = code
        self.alt_name = alt_name
        self.alt_code = alt_code
        self.model_name = model_name if model_name is not None else name
        self.model_code = model_code if model_code is not None else code
        self.index = index
        self.include = include  # include in output?

    def __str__(self):
        return f"ClassInfo: name={self.name}, code={self.code}, include={self.include}"
