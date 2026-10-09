# loki-geometry documentation

A documentation website (Sphinx, Read the Docs theme), made from the README and the code, so that it does not need
separate maintenance:

- the pages of the guide are the sections of `README.md` (split at its `##` headings by `source/conf.py`, which also
  rewrites the links between the sections; a new README section needs a line in `README_PAGES` of `conf.py`);
- "Command-line tools" are the help texts of the tools (`<tool> -h`);
- "Python API" shows the modules, classes and functions with their docstrings (`source/api.md`).

Build:

```sh
python3 -m pip install -r doc/requirements.txt   # once (and numpy)
cd doc
make html SPHINXOPTS=-W                          # warnings are errors
open build/html/index.html
```
