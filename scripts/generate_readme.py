#!/usr/bin/env python3

"""
Generate separate command and API reference files:
  - command-reference.md: Command Reference (from click --help output for all commands/subcommands)
  - api-reference.md: API Reference (from docstrings of the public API exported in __init__.__all__)

Usage examples:
  python scripts/generate_readme.py \
    --package britekit \
    --cli-module britekit.cli \
    --cli-object cli
"""

from __future__ import annotations
import argparse
import importlib
import inspect
import io
import os
import re
import sys
import textwrap
from typing import Any, Iterable, List, Tuple
from dataclasses import is_dataclass as dc_is_dataclass, fields as dc_fields, MISSING as DC_MISSING
from typing import get_origin, get_args

import itertools

SECTION_HEADINGS = (
    "Args",
    "Arguments",
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
)

def _md_escape(s: str) -> str:
    return s.replace("*", r"\*").replace("_", r"\_").replace("`", r"\`")

def _anchor_from_heading(text: str) -> str:
    """
    Create a GitHub-style anchor from a heading: lowercased, spaces -> '-',
    remove non-alphanumeric/hyphen chars, collapse multiple hyphens, trim hyphens.
    """
    s = text.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s

def _format_items_as_md_list(items):
    """
    items: list of dicts with keys: name, type (optional), default (optional), desc (optional)
    """
    lines = []
    for it in items:
        name = f"`{it['name']}`"
        typ = f" *({it['type']})*" if it.get("type") else ""
        default = f" *(default: {it['default']})*" if it.get("default") else ""
        desc = f" — {it.get('desc','').strip()}" if it.get("desc") else ""
        lines.append(f"- {name}{typ}{default}{desc}")
    return "\n".join(lines)

def _split_sections_google(text: str):
    """
    Very lightweight splitter for Google-style sections.
    Returns dict(section_name -> raw_section_text)
    """
    lines = text.splitlines()
    sections = {}
    cur = []
    cur_name = None
    for line in lines:
        if line.strip().rstrip(":") in SECTION_HEADINGS and line.strip().endswith(":"):
            if cur_name:
                sections[cur_name] = "\n".join(cur).strip()
            cur_name = line.strip().rstrip(":")
            cur = []
        else:
            cur.append(line)
    if cur_name:
        sections[cur_name] = "\n".join(cur).strip()
    return sections

def _parse_google_args(raw: str):
    """
    Parse lines like:
        name (Type): description...
        name: description...
    Handles hanging indentation for multi-line descriptions.
    """
    items = []
    cur = None
    for line in raw.splitlines():
        if not line.strip():
            if cur:
                items.append(cur); cur = None
            continue
        # new item?
        m = re.match(r"\s*([A-Za-z0-9_]+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$", line)
        if m:
            if cur: items.append(cur)
            cur = {"name": m.group(1), "type": m.group(2) or "", "desc": m.group(3).strip()}
        else:
            # continuation
            if cur:
                cur["desc"] = (cur["desc"] + " " + line.strip()).strip()
            else:
                # stray text—skip or attach to last
                pass
    if cur: items.append(cur)
    return items

def _parse_numpy_params(raw: str):
    """
    NumPy style:
        name : type
            description...
    """
    items = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", line)
        if m:
            name, typ = m.group(1), m.group(2)
            desc_lines = []
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith((" ", "\t"))):
                desc_lines.append(lines[i].strip())
                i += 1
            items.append({"name": name, "type": typ.strip(), "desc": " ".join(desc_lines).strip()})
            continue
        i += 1
    return items

def _parse_rest_params(text: str):
    """
    reST:
        :param name: description
        :type name: Type
    """
    descs = {}
    types = {}
    for line in text.splitlines():
        mp = re.match(r"\s*:param\s+([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        mt = re.match(r"\s*:type\s+([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        if mp:
            k = mp.group(1); descs[k] = descs.get(k, "") + " " + mp.group(2).strip()
        elif mt:
            k = mt.group(1); types[k] = mt.group(2).strip()
    items = []
    for k in sorted(set(descs) | set(types)):
        items.append({"name": k, "type": types.get(k, ""), "desc": descs.get(k, "").strip()})
    return items

def _reformat_docstring_to_markdown(doc: str) -> str:
    """
    Converts common docstring sections into Markdown lists where appropriate.
    Only modifies sections we recognize; leaves the rest intact.
    """
    if not doc:
        return ""

    # Normalize indentation
    doc = textwrap.dedent(doc).strip()

    # Try Google/NumPy first by detecting recognizable section headers
    sections = _split_sections_google(doc)
    if sections:
        out = doc  # start with original
        # Args/Parameters
        for key in ("Args", "Arguments", "Parameters"):
            if key in sections and sections[key]:
                raw = sections[key]
                # NumPy heading often looks like:
                # Parameters\n----------\n...
                if re.search(r"^-{3,}\s*$", raw.splitlines()[0] if raw.splitlines() else ""):
                    # Already consumed a dashed line; drop it
                    raw2 = "\n".join(raw.splitlines()[1:])
                    items = _parse_numpy_params(raw2)
                else:
                    # Heuristic: NumPy blocks often follow a dashed line *inside* the section too
                    if "----" in raw:
                        items = _parse_numpy_params(raw)
                        if not items:
                            items = _parse_google_args(raw)
                    else:
                        items = _parse_google_args(raw)

                if items:
                    md_list = _format_items_as_md_list(items)
                    # Replace the section in the doc
                    pattern = rf"(?ms)^{key}:\s*\n(.*?)(?=^\w.+:|\Z)"
                    out = re.sub(pattern, f"{key}:\n\n{md_list}\n\n", out)
        # Returns / Yields / Raises: turn into simple bullets per line
        for key in ("Returns", "Yields", "Raises"):
            if key in sections and sections[key]:
                raw = sections[key]
                # Try to parse NumPy-like returns: "name : type" lines
                items = _parse_numpy_params(raw)
                if not items:
                    # Fallback: bulletize non-empty lines
                    payload = "\n".join(f"- {line.strip()}" for line in raw.splitlines() if line.strip())
                else:
                    payload = _format_items_as_md_list(items)
                pattern = rf"(?ms)^{key}:\s*\n(.*?)(?=^\w.+:|\Z)"
                out = re.sub(pattern, f"{key}:\n\n{payload}\n\n", out)
        return out

    # reST fallback
    if ":param " in doc or ":type " in doc:
        items = _parse_rest_params(doc)
        if items:
            # Append a synthesized section
            md = _format_items_as_md_list(items)
            return doc + "\n\nParameters:\n\n" + md + "\n"
    return doc

def md_h2(text: str) -> str:
    return f"## {text}\n"

def md_h3(text: str) -> str:
    return f"### {text}\n"

def md_codeblock(code: str, lang: str = "") -> str:
    return f"```{lang}\n{code.rstrip()}\n```\n"

def dedent(s: str | None) -> str:
    if not s:
        return ""
    return textwrap.dedent(s).strip()

def format_signature(obj: Any) -> str:
    try:
        if inspect.isclass(obj):
            # Prefer __init__ signature; fall back to class signature
            init = getattr(obj, "__init__", None)
            if init is not None:
                sig = str(inspect.signature(init))
                # Replace self in signature for readability
                sig = re.sub(r"^\(self(?:, )?", "(", sig)
                return f"{obj.__name__}{sig}"
            return f"{obj.__name__}{inspect.signature(obj)}"
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            return f"{obj.__name__}{inspect.signature(obj)}"
        else:
            return obj.__name__  # best effort
    except Exception:
        # Some builtins/objects can raise
        return getattr(obj, "__name__", repr(obj))

def is_public_name(name: str) -> bool:
    return not name.startswith("_")

def public_methods(cls: type) -> List[Tuple[str, Any]]:
    items = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if is_public_name(name) and getattr(member, "__qualname__", "").startswith(cls.__name__ + "."):
            items.append((name, member))
    # Also consider @property getters that are public
    for name, member in inspect.getmembers(cls):
        if isinstance(member, property) and is_public_name(name):
            items.append((name, member))
    # Stable order: methods/properties by name
    return sorted(items, key=lambda kv: kv[0])

def render_api_for_object(obj_name: str, obj: Any) -> str:
    out = io.StringIO()
    if inspect.isclass(obj):
        out.write(md_h3(obj_name))
        out.write("**Class**  \n")
        out.write(md_codeblock(format_signature(obj), lang="python"))
        doc = dedent(_reformat_docstring_to_markdown(inspect.getdoc(obj)))
        if doc:
            out.write(doc + "\n\n")
        # Methods/properties
        methods = public_methods(obj)
        if methods:
            out.write("**Public methods & properties**\n\n")
            for mname, member in methods:
                out.write(f"**{mname}**  \n")

                if isinstance(member, property):
                    out.write(md_codeblock(f"@property {obj.__name__}.{mname}", lang="python"))
                    mdoc = _reformat_docstring_to_markdown(inspect.getdoc(member.fget) if member.fget else "")
                else:
                    out.write(md_codeblock(f"{obj.__name__}.{format_signature(member)}", lang="python"))
                    mdoc = _reformat_docstring_to_markdown(inspect.getdoc(member))
                if mdoc:
                    out.write(mdoc + "\n\n")
    elif inspect.isfunction(obj):
        out.write(md_h3(obj_name))
        out.write("**Function**  \n")
        out.write(md_codeblock(format_signature(obj), lang="python"))
        doc = dedent(inspect.getdoc(obj))
        if doc:
            out.write(doc + "\n\n")
    else:
        # Skip constants/vars; include doc if available
        doc = dedent(inspect.getdoc(obj))
        if doc:
            out.write(md_h3(obj_name))
            out.write(doc + "\n\n")
    return out.getvalue()

def collect_public_api(package_name: str) -> List[Tuple[str, Any]]:
    pkg = importlib.import_module(package_name)
    names: Iterable[str]
    if hasattr(pkg, "__all__") and isinstance(pkg.__all__, (list, tuple)):
        names = pkg.__all__
    else:
        names = [n for n in dir(pkg) if is_public_name(n)]
    result = []
    for name in sorted(set(names)):
        try:
            obj = getattr(pkg, name)
        except Exception:
            continue
        result.append((name, obj))
    return result

def render_api_section(package_name: str) -> str:
    parts = [md_h2("API Reference")]
    api_items = collect_public_api(package_name)

    # Build a class TOC at the top
    class_items: List[str] = []
    for name, obj in api_items:
        try:
            if inspect.isclass(obj):
                if name == "BaseConfig":
                    continue  # skip config container in the general API
                anchor = _anchor_from_heading(name)
                class_items.append(f"- [{name}](#{anchor})")
        except Exception:
            continue

    if class_items:
        parts.append(md_h3("Classes"))
        parts.append("\n".join(class_items) + "\n\n")

    # Render each object section
    for name, obj in api_items:
        if name == "BaseConfig":
            continue  # exclude from general API; documented separately
        parts.append(render_api_for_object(name, obj))
    return "".join(parts)

# ------------------------
# Config dataclass extraction
# ------------------------

def _friendly_type_name(t: Any) -> str:
    try:
        origin = get_origin(t)
        if origin is None:
            return getattr(t, "__name__", str(t))
        args = ", ".join(_friendly_type_name(a) for a in get_args(t))
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{args}]"
    except Exception:
        return str(t)

def _format_default(val: Any) -> str:
    try:
        if val is DC_MISSING:
            return "—"
        if callable(val):
            # default_factory token
            return f"<factory {getattr(val, '__name__', 'callable')}>"
        return repr(val)
    except Exception:
        return str(val)

def render_config_reference() -> str:
    """
    Generate a focused configuration reference for dataclasses defined in
    britekit.core.base_config: Audio, Training, Inference, Miscellaneous, BaseConfig.
    """
    import importlib
    import importlib.util
    mod = importlib.import_module("britekit.core.base_config")

    dataclass_names = [
        "Audio",
        "Training",
        "Inference",
        "Miscellaneous",
        "BaseConfig",
        "FunctionConfig",
    ]

    parts: List[str] = [md_h2("Configuration Reference")]

    # Build descriptions by parsing comments in source file
    def _load_source_text() -> str:
        try:
            spec = importlib.util.find_spec("britekit.core.base_config")
            if spec and getattr(spec, "origin", None):
                with open(spec.origin, "r", encoding="utf-8") as fh:
                    return fh.read()
        except Exception:
            pass
        return ""

    def _parse_descriptions_from_source(src_text: str) -> dict[str, dict[str, str]]:
        """
        Return mapping: class_name -> { field_name -> description }
        Pulls from inline comments and immediately preceding comment lines.
        """
        result: dict[str, dict[str, str]] = {name: {} for name in dataclass_names}
        if not src_text:
            return result
        lines = src_text.splitlines()
        # Index class name to (start, end)
        indices: dict[str, tuple[int, int]] = {}
        # Find class starts
        class_positions: List[tuple[str, int]] = []
        for idx, line in enumerate(lines):
            m = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$", line)
            if m:
                cname = m.group(1)
                if cname in dataclass_names:
                    class_positions.append((cname, idx))
        # Determine ends as next class start or EOF
        for i, (cname, start) in enumerate(class_positions):
            end = len(lines)
            if i + 1 < len(class_positions):
                end = class_positions[i + 1][1]
            indices[cname] = (start, end)
        # For each class, scan fields
        for cname, (start, end) in indices.items():
            # Determine class indent (0) and field indent as any indent > 0
            for i in range(start + 1, end):
                line = lines[i]
                # Match a field line like: "    name: Type = default  # comment"
                m = re.match(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*.+$", line)
                if not m:
                    continue
                indent, fname = m.group(1), m.group(2)
                # Skip dunder or uppercase constants
                if fname.startswith("__"):
                    continue
                # Inline comment
                desc = ""
                try:
                    # crude split on # for inline comment
                    if "#" in line:
                        after_hash = line.split("#", 1)[1].strip()
                        if after_hash:
                            desc = after_hash
                except Exception:
                    pass
                # If no inline, gather preceding comment lines
                if not desc:
                    j = i - 1
                    comment_lines: List[str] = []
                    while j > start:
                        prev = lines[j].rstrip()
                        if prev.strip() == "":
                            if comment_lines:
                                break
                            j -= 1
                            continue
                        cm = re.match(r"^\s*#\s?(.*)$", prev)
                        if cm:
                            comment_lines.append(cm.group(1).strip())
                            j -= 1
                            continue
                        break
                    if comment_lines:
                        desc = " ".join(reversed([c for c in comment_lines if c]))
                if desc:
                    result.setdefault(cname, {})[fname] = desc
        return result

    descriptions = _parse_descriptions_from_source(_load_source_text())

    # Top-level TOC
    parts.append(md_h3("Sections"))
    parts.append("\n".join(f"- [{name}](#{name.lower()})" for name in dataclass_names) + "\n\n")

    for name in dataclass_names:
        obj = getattr(mod, name, None)
        if obj is None or not dc_is_dataclass(obj):
            continue
        parts.append(md_h3(name))
        # Table header
        rows: List[str] = ["| Field | Type | Default | Description |", "| --- | --- | --- | --- |"]
        for f in dc_fields(obj):
            ftype = _friendly_type_name(f.type)
            default = None
            if f.default is not DC_MISSING:
                default = f.default
            elif f.default_factory is not DC_MISSING:  # type: ignore[attr-defined]
                default = f.default_factory  # type: ignore[attr-defined]
            else:
                default = DC_MISSING
            desc = descriptions.get(name, {}).get(f.name, "")
            # Escape pipes in desc
            if "|" in desc:
                desc = desc.replace("|", r"\|")
            rows.append(f"| `{f.name}` | `{ftype}` | { _format_default(default) } | {desc} |")
        parts.append("\n".join(rows) + "\n\n")

    return "".join(parts)

# ------------------------
# Click help extraction
# ------------------------

def resolve_cli(cli_module: str, cli_object: str) -> Any:
    mod = importlib.import_module(cli_module)
    try:
        return getattr(mod, cli_object)
    except AttributeError as e:
        raise SystemExit(f"Could not find '{cli_object}' in module '{cli_module}'.") from e

def collect_click_commands(root_command: Any) -> List[Tuple[str, Any]]:
    """
    Depth-first list of (command_path, command_obj).
    Supports nested click.Group trees.
    """
    import click  # local import
    commands: List[Tuple[str, Any]] = []

    def visit(path: List[str], cmd: Any):
        commands.append((" ".join(path) if path else "", cmd))
        if isinstance(cmd, click.Group):
            for subname in sorted(cmd.commands):
                visit(path + [subname], cmd.commands[subname])

    visit([], root_command)
    return commands

def render_command_reference(cli_module: str, cli_object: str) -> str:
    from click.testing import CliRunner
    runner = CliRunner()
    root = resolve_cli(cli_module, cli_object)
    parts = [md_h2("Command Reference")]

    all_cmds = collect_click_commands(root)

    # Precompute anchors for all command sections
    anchor_map = {}
    for full_name, _ in all_cmds:
        title = "britekit" if full_name == "" else f"britekit {full_name}"
        anchor_map[full_name] = _anchor_from_heading(title)

    # Build a Markdown commands table with links to each section
    table_lines: List[str] = ["| Command | Description |", "| --- | --- |"]
    for full_name, cmd in all_cmds:
        if full_name == "":
            continue  # omit root row to avoid linking to a removed section
        title = f"britekit {full_name}"
        anchor = anchor_map.get(full_name, _anchor_from_heading(title))
        # Short description
        desc = getattr(cmd, "short_help", None) or (getattr(cmd, "help", None) or "").strip().split("\n")[0]
        table_lines.append(f"| [{title}](#{anchor}) | {desc} |")
    parts.append("\n".join(table_lines) + "\n\n")

    def _link_commands_table(help_text: str, current_path: str) -> str:
        lines = help_text.splitlines()
        in_commands = False
        for i, line in enumerate(lines):
            if not in_commands:
                if re.match(r"^\s*Commands\s*:\s*$", line):
                    in_commands = True
                continue
            # inside Commands: block; stop on blank line or next section header
            if not line.strip():
                in_commands = False
                continue
            # Match typical Click table row: indent, name, 2+ spaces, desc
            m = re.match(r"^(\s*)([A-Za-z0-9_.:-]+)(\s{2,}.*)$", line)
            if m:
                indent, cmdname, rest = m.groups()
                full = (current_path + (" " if current_path and cmdname else "") + cmdname).strip()
                anchor = anchor_map.get(full)
                if anchor:
                    lines[i] = f"{indent}[{cmdname}](#{anchor}){rest}"
                continue
            # If the line doesn't match a row, consider the section ended when a non-indented header appears
            if re.match(r"^\S", line):
                in_commands = False
        return "\n".join(lines)
    # Ensure root help first
    for full_name, cmd in all_cmds:
        if full_name == "":
            continue  # skip root section (was previously britekit --help output)
        args = [] if full_name == "" else full_name.split(" ")
        result = runner.invoke(root, args + ["--help"])
        help_text = result.output.strip()
        help_text = help_text.replace("Usage: cli", "Usage: britekit")
        help_text = _link_commands_table(help_text, full_name)
        title = f"britekit {full_name}"
        parts.append(md_h3(title))
        # Exact match of --help, preserved
        parts.append(md_codeblock(help_text, lang=""))
    return "".join(parts)

# ------------------------
# Command API extraction
# ------------------------

def _find_impl_function_from_callback(cb: Any) -> Any | None:
    """
    Best-effort: given a Click callback function, try to locate a corresponding
    implementation function named like "<base>_impl" in the same module by
    scanning the callback source for a call to *_impl(...).
    """
    try:
        import inspect as _inspect
        import importlib as _importlib
        src = _inspect.getsource(cb)
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)_impl\s*\(", src)
        if not m:
            return None
        impl_name = f"{m.group(1)}_impl"
        mod = _importlib.import_module(cb.__module__)
        return getattr(mod, impl_name, None)
    except Exception:
        return None

def render_command_api(commands_module: str) -> str:
    """
    Render a Command API Reference by walking the public objects in the
    commands module. For Click commands, we try to surface the underlying
    implementation function's signature and docstring when available.
    """
    import importlib
    import click  # type: ignore

    mod = importlib.import_module(commands_module)
    # Determine which names to include: prefer module __all__, else public attrs
    if hasattr(mod, "__all__") and isinstance(mod.__all__, (list, tuple)):
        names: Iterable[str] = mod.__all__  # type: ignore[assignment]
    else:
        names = [n for n in dir(mod) if is_public_name(n)]

    parts = [md_h2("Command API Reference")]
    for name in sorted(set(names)):
        try:
            obj = getattr(mod, name)
        except Exception:
            continue

        # Prefer Click commands
        if isinstance(obj, click.core.Command):  # type: ignore[attr-defined]
            # Render under the non-impl function name, but pull docs from *_impl if present
            cb = None
            try:
                cb = obj.callback
            except Exception:
                cb = None

            if cb is not None:
                impl_fn = _find_impl_function_from_callback(cb)
                # Section header
                parts.append(md_h3(name))
                parts.append("**Function**  \n")
                # Prefer showing the non-impl function signature
                parts.append(md_codeblock(format_signature(cb), lang="python"))
                # Prefer impl docstring for rich descriptions; fallback to callback doc/help
                doc = _reformat_docstring_to_markdown(inspect.getdoc(impl_fn) if impl_fn else (inspect.getdoc(cb) or ""))
                if not doc:
                    help_text = obj.help or ""
                    doc = dedent(help_text)
                if doc:
                    parts.append(doc + "\n\n")
            else:
                # Fallback: render the Click command help as documentation
                parts.append(md_h3(name))
                parts.append("**Command**  \n")
                help_text = obj.help or ""
                if help_text:
                    parts.append(dedent(help_text) + "\n\n")
            continue

        # If this is a plain function/class, render as-is
        if inspect.isfunction(obj) or inspect.isclass(obj):
            parts.append(render_api_for_object(name, obj))
            continue

        # Otherwise, skip (constants, modules, etc.) unless they have a docstring
        doc = dedent(inspect.getdoc(obj))
        if doc:
            parts.append(md_h3(name))
            parts.append(doc + "\n\n")

    return "".join(parts)

# ------------------------
# File generation
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate separate command and API reference files.")
    ap.add_argument("--package", default="britekit", help="Python package to document (e.g., britekit)")
    ap.add_argument("--cli-module", default="britekit.cli", help="Module that defines the click CLI (default: britekit.cli)")
    ap.add_argument("--cli-object", default="cli", help="Name of the click Group object (default: cli)")
    ap.add_argument("--command-output", default="command-reference.md", help="Output file for command reference (default: command-reference.md)")
    ap.add_argument("--api-output", default="api-reference.md", help="Output file for API reference (default: api-reference.md)")
    ap.add_argument("--config-output", default="config-reference.md", help="Output file for configuration reference (default: config-reference.md)")
    ap.add_argument("--commands-module", default="britekit.commands", help="Module that exposes public command callables (default: britekit.commands)")
    ap.add_argument("--command-api-output", default="command-api-reference.md", help="Output file for command API reference (default: command-api-reference.md)")
    args = ap.parse_args()

    # Import-time side effects: if your package has heavy imports, run in a controlled env or lazy-import pattern.
    api_md = render_api_section(args.package)
    cmd_md = render_command_reference(args.cli_module, args.cli_object)
    cmd_api_md = render_command_api(args.commands_module)
    cfg_md = render_config_reference()

    # Write command reference file
    with open(args.command_output, "w", encoding="utf-8") as f:
        f.write(cmd_md.strip() + "\n")

    # Write API reference file
    with open(args.api_output, "w", encoding="utf-8") as f:
        f.write(api_md.strip() + "\n")

    # Write configuration reference file
    with open(args.config_output, "w", encoding="utf-8") as f:
        f.write(cfg_md.strip() + "\n")

    # Write Command API reference file
    with open(args.command_api_output, "w", encoding="utf-8") as f:
        f.write(cmd_api_md.strip() + "\n")

    print(f"Wrote {args.command_output}")
    print(f"Wrote {args.api_output}")
    print(f"Wrote {args.config_output}")
    print(f"Wrote {args.command_api_output}")

if __name__ == "__main__":
    # Ensure project root on sys.path if run from repo root
    sys.path.insert(0, os.path.abspath("."))
    # And ensure src/ is on sys.path for src-layout projects
    sys.path.insert(0, os.path.abspath("src"))
    main()
