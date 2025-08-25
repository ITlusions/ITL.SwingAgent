# Additional files that might be useful

## .gitignore entries for docs
docs/site/
site/

## Requirements for docs build
# Add to pyproject.toml:
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0", 
    "mkdocs-mermaid2-plugin>=0.6.0",
    "pymdown-extensions>=10.0.0"
]

## Build commands
# Install docs dependencies:
# pip install -e ".[docs]"

# Serve locally:
# mkdocs serve

# Build static site:
# mkdocs build

# Deploy to GitHub Pages:
# mkdocs gh-deploy