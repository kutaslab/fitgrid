# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = fitgrid
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile api-rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

api-rst: 
	sphinx-apidoc -e -f -o source/api_docs ../fitgrid
	awk 'BEGIN{printf(":orphan:\n\n")}{print}' source/api_docs/modules.rst > source/api_docs/modules.tmp
	mv source/api_docs/modules.tmp source/api_docs/modules.rst
	ls source/api_docs

# just format .rst, don't run sphinx-gallery scripts
html-noexec: api-rst
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# cleaning auto_gallery triggers re-running all sphinx-gallery scripts
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf source/api_docs/*
	rm -rf source/auto_gallery/*

