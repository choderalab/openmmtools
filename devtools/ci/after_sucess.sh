echo $TRAVIS_PULL_REQUEST $TRAVIS_BRANCH

if [[ "$TRAVIS_PULL_REQUEST" == "true" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi


if [[ "$TRAVIS_BRANCH" != "master" ]]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi


if [[ "2.7 3.3 3.4" =~ "$python" ]]; then
    conda install --yes --quiet binstar
    BINSTAR_VERSION=`git describe 2> /dev/null || git rev-parse --short HEAD` # From http://wygoda.net/blog/getting-useful-git-revision-information/
    binstar -t $BINSTAR_TOKEN upload -v $BINSTAR_VERSION -u omnia -p openmmtools-dev $HOME/miniconda/conda-bld/linux-64/${PACKAGENAME}-*
fi

if [[ "$python" != "2.7" ]]; then
    echo "No deploy on PYTHON_VERSION=${python}"; exit 0
fi

