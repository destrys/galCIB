# Contributing

## Installing galCIB for Development

For developing `galCIB`, we use [`uv`](https://docs.astral.sh/uv/).
If you aren't updating dependencies, it is possible to use other tools
(e.g. conda or venv) to write and test code, but `uv` makes managing
python versions and dependencies much simpler and more predicable.

Install `uv` using the instructions at
[https://docs.astral.sh/uv/getting-started/installation/]

Clone this repo:
```
git clone https://github.com/tanveerkarim/galCIB.git
cd galCIB
uv sync --dev
```

`uv sync --dev` will download the pinned version of python if you
don't have it, install dependencies (with the dev dependencies),
and create a venv for you.
From there, you can use `uv run` to run commands in the venv without having
to activate/deactive.

So to run the tests:
```
uv run pytest
```

or to run a tutorial jupyter notebook:
```
uv run jupyter lab
```

If you don't want to type `uv run`, you can activate the venv with
```
source .venv/bin/activate
```

### Common workflows

#### Only Modifying Code

- Install repo via the instructions above.
- Modify the code
- Run tests: `uv run pytest`
- Submit PR via GitHub

#### Adding a (dev) dependency

- Install repo via the instructions above
- Add a new dependecy with `uv add package_name`
- If it's a dev dependency (a dependency only used for development, e.g. `pytest`) add the `--dev` flag: `uv add --dev package_name`
- If you want to pin the dependency, you can set: `uv add "package_name==1.2.3"`, otherwise the newest version that is compatible with the other dependencies will be used.
- Commit the `uv.lock` file when you commit changes to git
- See [https://docs.astral.sh/uv/concepts/projects/dependencies/] for more details

#### Upgrading or modifying a dependency

Upgrading/modifying a dependency is just adding the dependency again while setting the
`--upgrade-package <name>` option.

- Install repo via the instructions above
- `uv add "package_name==new_version" --upgrade-package package_name`
- Commit the `uv.lock` file when you commit changes to git
- See [https://docs.astral.sh/uv/concepts/projects/dependencies/] for more details

#### Installing repo from editable source in another uv-managed project

If you want to install this package (from editable local files)
in another uv-managed project
please see [https://docs.astral.sh/uv/concepts/projects/dependencies/#path]
for how to add a project at a local path. 