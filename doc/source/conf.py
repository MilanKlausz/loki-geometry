"""Sphinx configuration of the loki-geometry documentation.

The README is the single source of the text: at each build its sections are written as pages to generated/
(split at the "## " headings, links between the sections rewritten, relative file links pointing to GitHub), and
the help texts of the command-line tools are collected in generated/cli.md. The API reference (api.md) comes from
the docstrings. Build: make html in doc/ (see doc/README.md).
"""

import contextlib
import io
import os
import pathlib
import re
import sys

SOURCE = pathlib.Path(__file__).parent.resolve()
REPO = SOURCE.parent.parent
GENERATED = SOURCE / "generated"
GITHUB = "https://github.com/MilanKlausz/loki-geometry"
GITHUB_BRANCH = "main"
sys.path.insert(0, str(REPO / "src"))

import loki_geometry  # noqa: E402

# -- Project information -----------------------------------------------------

project = "loki-geometry"
author = "Milan Klausz"
copyright = "Milan Klausz"
release = loki_geometry.__version__
version = release

# -- Pages from the README -----------------------------------------------------

# page name: (page title, the README sections on it); a page with one section of the same title gets its text
README_PAGES = {
    "installation": ("Installation", ["Installation"]),
    "updating": ("Updating the instrument JSON file", ["Updating the instrument JSON file"]),
    "pixel_json": (
        "Pixel geometry for the instrument JSON file",
        ["Pixel geometry for the instrument JSON file"],
    ),
    "calibration": (
        "From the measured points to the 8 corner positions",
        ["From the measured points to the 8 corner positions"],
    ),
    "new_survey": ("A new survey", ["A new survey"]),
    "geometry_info": ("Geometry information", ["Geometry information"]),
    "geant4": (
        "Bank calibration for the Geant4 geometry",
        ["Bank calibration for the Geant4 geometry"],
    ),
    "conventions": ("Conventions", ["Conventions"]),
    "package": ("Package contents", ["Package contents"]),
    "other_commands": ("Other commands", ["Other commands"]),
    "development": (
        "Development and known issues",
        ["Development", "Open points / known issues", "License"],
    ),
}


def slug(title):
    """The anchor of a heading (as GitHub and MyST make it)."""
    text = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return text.replace(" ", "-")


def readme_sections(text):
    """(intro, {section title: section text}) of the README, split at the "## " headings outside code blocks."""
    intro, sections, current, fence = [], {}, None, False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current is None:
            if not line.startswith("# "):
                intro.append(line)
        else:
            sections[current].append(line)
    return "\n".join(intro).strip() + "\n", {
        k: "\n".join(v).strip() + "\n" for k, v in sections.items()
    }


def promote_headings(text):
    """### -> ## etc. (outside code blocks): a section that becomes a page with its own title."""
    lines, fence = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"#{3,} ", line):
            line = line[1:]
        lines.append(line)
    return "\n".join(lines) + "\n"


def rewrite_links(text, page_of_anchor, page, doc_prefix=None):
    """README links -> links of the documentation: #anchor of another page -> page.md#anchor, a relative file
    link -> GitHub. doc_prefix (for the intro, included in index.md): links to pages as {doc} roles with this
    path prefix (they resolve relative to the including page)."""

    def replace(match):
        label, target = match.group(1), match.group(2)
        if target.startswith("#"):
            target_page = page_of_anchor.get(target[1:])
            if target_page is None:
                raise ValueError(f"README link to an unknown section: {target}")
            if target_page == page:
                return f"[{label}]({target})"
            anchor = "" if target[1:] == slug(README_PAGES[target_page][0]) else target
            if doc_prefix is not None:
                if anchor:
                    raise ValueError(f"README intro: links to sections are not supported: {target}")
                return f"{{doc}}`{label} <{doc_prefix}{target_page}>`"
            return f"[{label}]({target_page}.md{anchor})"
        if re.match(r"[a-z]+:", target):
            return match.group(0)
        kind = "tree" if (REPO / target).is_dir() else "blob"
        return f"[{label}]({GITHUB}/{kind}/{GITHUB_BRANCH}/{target})"

    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", replace, text)


def generate_readme_pages():
    intro, sections = readme_sections((REPO / "README.md").read_text())
    missing = set(sections) - {s for _, secs in README_PAGES.values() for s in secs}
    if missing:
        raise ValueError(f"README sections without a page in doc/source/conf.py: {sorted(missing)}")
    page_of_anchor = {}
    for page, (title, secs) in README_PAGES.items():
        page_of_anchor[slug(title)] = page
        for section in secs:
            page_of_anchor[slug(section)] = page
            for line in sections[section].splitlines():
                if re.match(r"#{3,} ", line):
                    page_of_anchor[slug(line.lstrip("#"))] = page
    GENERATED.mkdir(exist_ok=True)
    note = "<!-- generated from README.md by doc/source/conf.py: edit the README -->\n\n"
    for page, (title, secs) in README_PAGES.items():
        if secs == [title]:
            body = promote_headings(sections[title])
        else:
            body = "\n".join(f"## {s}\n\n{sections[s]}" for s in secs)
        text = f"{note}# {title}\n\n{rewrite_links(body, page_of_anchor, page)}"
        write_if_changed(GENERATED / f"{page}.md", text)
    write_if_changed(
        GENERATED / "intro.md", rewrite_links(intro, page_of_anchor, None, "generated/")
    )


def generate_cli_page():
    """The help texts of the command-line tools (the console scripts of pyproject.toml)."""
    import importlib

    scripts = re.findall(
        r'^(loki-[\w-]+) = "([\w.]+):(\w+)"', (REPO / "pyproject.toml").read_text(), re.M
    )
    parts = [
        "# Command-line tools\n",
        "The help texts of the tools (`<tool> -h`), generated from the code.\n",
    ]
    old_argv, old_columns = sys.argv, os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "86"  # fits the page width
    try:
        for name, module, function in sorted(scripts):
            sys.argv = [name]
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.suppress(SystemExit):
                getattr(importlib.import_module(module), function)(["-h"])
            parts.append(f"## {name}\n\n```text\n{output.getvalue().rstrip()}\n```\n")
    finally:
        sys.argv = old_argv
        if old_columns is None:
            del os.environ["COLUMNS"]
        else:
            os.environ["COLUMNS"] = old_columns
    write_if_changed(GENERATED / "cli.md", "\n".join(parts))


def write_if_changed(path, text):
    """Keeps the file time of unchanged pages, so that Sphinx rebuilds only the changed ones."""
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


generate_readme_pages()
generate_cli_page()

# -- General configuration ---------------------------------------------------

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.viewcode"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
exclude_patterns = ["build", "generated/intro.md"]  # intro.md: included in index.md
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"loki-geometry {release}"
html_context = {
    "display_github": True,
    "github_user": "MilanKlausz",
    "github_repo": "loki-geometry",
    "github_version": GITHUB_BRANCH,
    "conf_py_path": "/doc/source/",
}
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_with_keys": True,
    "navigation_depth": 3,
}


def _github_url_of_generated_pages(app, pagename, templatename, context, doctree):
    """'Edit on GitHub' of the pages made from the README: the README."""
    if pagename.startswith("generated/"):
        target = "README.md" if pagename != "generated/cli" else "pyproject.toml"
        context["meta"] = dict(
            context.get("meta") or {}, github_url=f"{GITHUB}/blob/{GITHUB_BRANCH}/{target}"
        )


def _docstring_as_text(app, what, name, obj, options, lines):
    """The docstrings are plain text (not reStructuredText): show them as preformatted text."""
    if any(line.strip() for line in lines):
        lines[:] = (
            [".. code-block:: text", ""]
            + [f"    {line}" if line.strip() else "" for line in lines]
            + [""]
        )


def setup(app):
    app.connect("autodoc-process-docstring", _docstring_as_text)
    app.connect("html-page-context", _github_url_of_generated_pages)
