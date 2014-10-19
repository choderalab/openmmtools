.. _getting-started:

Getting started
###############

Dependencies
============

``pymbar`` requires ``numpy`` and ``scipy``. You'll also need a working C
compiler and build environment, to compile various C-level extensions.

Easy Way (Recommended)
----------------------

The easiest way to get all of the dependencies is to install one of the 
pre-packaged scientific python distributes like `Enthought's Canopy 
<https://www.enthought.com/products/canopy/>`_ or `Continuum's Anaconda 
<https://store.continuum.io/>`_. These distributions already contain all of 
the dependences, and are distributed via 1-click installers.

Medium Way
----------

Linux
++++++
If you're on ubuntu and have root, you can install everything through your package manager. ::

  $ sudo apt-get install python-dev python-numpy python-nose python-setuptools python-scipy

Mac
+++
If you're on mac and want a package manager, you should be using `homebrew <http://mxcl.github.io/homebrew/>`_ and ``brews``'s python (see `this page <https://github.com/mxcl/homebrew/wiki/Homebrew-and-Python>`_ for details). The best way to install numpy and scipy with ``brew`` is to use
samueljohn's tap. ::

  $ brew tap samueljohn/python
  $ brew install python
  $ brew install numpy
  $ brew install scipy

Harder Way : Compiling from source (no root needed)
---------------------------------------------------

If you don't already have a python installation you want to use, you can compile a new one. ::

  $ wget http://www.python.org/ftp/python/2.7.5/Python-2.7.5.tgz
  $ tar -xzvf Python-2.7.5.tgz
  $ cd Python-2.7.5
  $ ./configure --prefix=$HOME/local/python
  $ make
  $ make install

  $ export PATH=$HOME/local/python/bin:$PATH

If you don't have ``easy_install`` or ``pip`` yet, you can get them with ::

  $ wget http://pypi.python.org/packages/source/s/setuptools/setuptools-0.6c11.tar.gz
  $ tar -xzvf setuptools-0.6c11.tar.gz
  $ cd setuptools-0.6c11.tar.gz
  $ python setup.py install
  $ easy_install pip

Now you're home free ::

  $ pip install numpy
  $ pip install scipy

Installing ``pymbar``
=====================

``pymbar`` currently runs best on Python 2.7.x; earlier versions of Python are not
supported.  ``pymbar`` is developed and
tested on mac and linux platforms. 

Easy Way (Recommended)
----------------------

Just run ::

  $ pip install pymbar

Medium Way (Advanced Users Only)
------------------------------------
To get the latest unstable version, clone the source code repository from github::

  $ git clone git://github.com/choderalab/pymbar.git

Then, in the directory containing the source code, you can install it with. ::

  $ python setup.py install


Running the tests
=================
Running the tests is a great way to verify that everything is working. The test
suite uses `nose <https://nose.readthedocs.org/en/latest/>`_, which you can pick
up via ``pip`` if you don't already have it. ::

  $ pip install nose

Then enter the ``openmm-testsystems`` the source directory and run ::

  $ nosetests

