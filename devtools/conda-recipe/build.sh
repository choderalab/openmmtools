#!/bin/bash

$PYTHON setup.py install

# Copy examples
# TODO: Have setup.py install examples instead?
mkdir $PREFIX/share/openmmtools
cp -r examples $PREFIX/share/openmmtools/
