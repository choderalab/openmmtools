#!/bin/bash
# Must be invoked with $PACKAGENAME

echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH
PUSH_DOCS_TO_S3=false

if [ "$TRAVIS_PULL_REQUEST" = true ]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi


if [ "$TRAVIS_BRANCH" != "master" ]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi


# Deploy to binstar
conda install --yes anaconda-client jinja2
pushd .
cd $HOME/miniconda/conda-bld
FILES=*/${PACKAGENAME}-dev-*.tar.bz2
for filename in $FILES; do
    echo "********************************************************************************"
    echo "Removing $filename from anaconda cloud..."
    echo "anaconda remove --force ${ORGNAME}/${PACKAGENAME}-dev/dev/${filename}"
    anaconda -t $BINSTAR_TOKEN remove --force ${ORGNAME}/${PACKAGENAME}-dev/dev/${filename}
    echo "********************************************************************************"
    echo "Uploading $filename to anaconda cloud..."
    echo "anaconda upload --force -u ${ORGNAME} -p ${PACKAGENAME}-dev ${filename}"
    anaconda -t $BINSTAR_TOKEN upload --force -u ${ORGNAME} -p ${PACKAGENAME}-dev ${filename}
    echo "********************************************************************************"
done
popd
