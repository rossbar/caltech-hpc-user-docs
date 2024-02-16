# Working with Caltech HPC - Docs

Collection of user-contributed documentation related to working with Caltech HPC.

## Contribute

### First-time setup

Make an environment with all the necessary dependencies to build the site.

```bash
$ python -m venv hpc-docs-env
$ source hpc-docs-env/bin/activate
$ pip install -r requirements.txt
```

### Build the site

The sphinx site can be built with `make html`.

The static site can then be viewed with `firefox _build/html/index.html`
