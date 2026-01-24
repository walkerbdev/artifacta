# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Build UI docs first, then Sphinx docs
html: jsdoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Copying JSDoc files to HTML output..."
	@cp -r $(BUILDDIR)/jsdoc $(BUILDDIR)/html/jsdoc

# Generate JSDoc documentation
jsdoc:
	@echo "Generating UI API documentation with JSDoc..."
	@cd .. && npm run docs:ui || true
	@echo "JSDoc documentation generated in $(BUILDDIR)/jsdoc/"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
