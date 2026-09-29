"""
Build the aframe API documentation with pdoc.

    pip install -e ".[plot,docs]"
    python docs/make_docs.py              # static site in ./site
    python docs/make_docs.py -o <dir>     # static site in <dir>

The landing page is the ``aframe`` package page: the README followed by the
public API. ``aframe.utils`` (meshing, load transfer and plotting) gets its own pages.
"""
import argparse
from pathlib import Path

import pdoc
import pdoc.render

import aframe
from aframe.utils import plot_pyvista

MODULES = ['aframe', 'aframe.utils']
SOURCE_URL = 'https://github.com/LSDOlab/aframe/blob/main/aframe/'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output-directory', default='site', type=Path)
    output = parser.parse_args().output_directory

    # aframe loads its plotting helpers lazily on first use, but pdoc only
    # documents attributes that already exist, so load them here
    for name in aframe._PLOTTING:
        setattr(aframe, name, getattr(plot_pyvista, name))

    pdoc.render.configure(docformat='numpy',
                          edit_url_map={'aframe': SOURCE_URL},
                          footer_text=f'aframe {aframe.__version__}')
    pdoc.pdoc(*MODULES, output_directory=output)

    # with several modules pdoc writes a module list as index.html; open the
    # package page (README + API) instead
    (output / 'index.html').write_text('<!doctype html>\n<meta http-equiv="refresh" content="0; url=aframe.html">\n'
                                       '<a href="aframe.html">aframe documentation</a>\n', encoding='utf-8')
    print(f'documentation written to {output.resolve()}')


if __name__ == '__main__':
    main()
