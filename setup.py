"""
Various Python tools for OpenMM.
"""
import distutils.extension
from Cython.Build import cythonize

import sys
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:]),

################################################################################
# SETUP
################################################################################

extensions = distutils.extension.Extension("openmmtools.multistate.mixing._mix_replicas",
                                           ['./openmmtools/multistate/mixing/_mix_replicas.pyx'])

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name='openmmtools',
    author='John Chodera',
    author_email='john.chodera@choderalab.org',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    url='https://github.com/choderalab/openmmtools',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    platforms=['Linux',
               'Mac OS-X',
               'Unix',
               'Windows'],            # Valid platforms your code works on, adjust to your flavor
    classifiers=CLASSIFIERS.splitlines(),
    python_requires=">=3.6",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    zip_safe=False,

    ext_modules=cythonize(extensions),
    entry_points={'console_scripts': [
        'test-openmm-platforms = openmmtools.scripts.test_openmm_platforms:main',
        'benchmark-alchemy = openmmtools.tests.test_alchemy:benchmark_alchemy_from_pdb',
        ]},
    )
