#!/bin/bash

cp -r $RECIPE_DIR/../.. $SRC_DIR
$PYTHON setup.py clean
$PYTHON setup.py install

# Copy examples
# TODO: Have setup.py install examples instead?
mkdir $PREFIX/share/openmmtools
cp -r $RECIPE_DIR/../../examples $PREFIX/share/openmmtools/
