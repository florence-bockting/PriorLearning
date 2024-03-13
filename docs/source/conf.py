# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PriorLearning'
copyright = '2024, Florence Bockting'
author = 'Florence Bockting'
release = '0.0.1'
#theme: `jekyll-theme-minimal`
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",     # Support automatic documentation
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",    # Automatically check if functions are documented
    "sphinx.ext.mathjax",     # Allow support for Latex
    "sphinx.ext.viewcode",    # Include the source code in documentation
    "sphinx.ext.githubpages", # Build for GitHub pages
    'sphinxemoji.sphinxemoji', # Allows emojis in RST files
    "numpydoc",               # Support NumPy style docstrings
    "myst_nb",                # For compiling Jupyter Notebooks into high quality documentation formats
    'sphinx.ext.autosectionlabel', # Using references to files in local folders   
    "sphinx_design"    
]
autoclass_content = 'both'

numpydoc_show_class_members = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ["http", "https", "mailto"]
autodoc_default_options = {
    "members": "var1, var2",
    "special-members": "__call__,__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "member-order": "bysource"
}

html_theme_options = {
    "logo": {
      "image_light": "_static/mmp-logo-light.png",
      "image_dark": "_static/mmp-logo-dark.png",
   },
   "repository_url": "https://github.com/florence-bockting/PriorLearning",
    "use_repository_button": True,
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = 'Prior Learning'
html_static_path = ['_static']
#html_logo = "_static/mmp-logo-light.png"
html_favicon = "_static/favicon-light.ico"

# do not execute jupyter notebooks when building docs
nb_execution_mode = "off"

# download notebooks as .ipynb and not as .ipynb.txt
html_sourcelink_suffix = ""

import os
suppress_warnings = [
    f"autosectionlabel._examples/{filename.split('.')[0]}"
    for filename in os.listdir("case_studies/")
    if os.path.isfile(os.path.join("case_studies/", filename))
]  # Avoid duplicate label warnings for Jupyter notebooks.

remove_from_toctrees = ["_autosummary/*"]