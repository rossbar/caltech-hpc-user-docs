# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Caltech HPC User Knowledge Base'
copyright = '2024, rossbar'
author = 'rossbar'
release = '2024.02.15-dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_togglebutton",
    "sphinx_design",
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', 'hpc-docs-env', "README.md"
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = "Caltech HPC User Guides"
#html_static_path = ['_static']
html_theme_options = {
    "github_url": "https://github.com/rossbar/caltech-hpc-user-docs",
    "repository_url": "https://github.com/rossbar/caltech-hpc-user-docs",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}
