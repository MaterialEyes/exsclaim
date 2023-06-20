Contributing to exsclaim!
==============================

If there is something you would like to add (or see added) to exsclaim,
your first step should be to check for an existing `issue <https://github.com/MaterialEyes/exsclaim/issues>`_.
If none exist for your bug/feature, create a new one. This way other people
will know you are working on it and be able to discuss.

Setup
------------------------------

It is recommended that you develop on Linux or Mac OSX. If you are using Windows,
using `WSL2 <https://docs.microsoft.com/en-us/windows/wsl/install>`_ is recommended
so you can access a linux kernel on Windows.

To start run the following commands::
    git clone git@github.com:MaterialEyes/exsclaim.git

Development Environment
------------------------------

To develop exsclaim, it is strongly recommended you use Docker. The easiest way
is to develop with a code editor that supports developing inside containers.
Visual Studio Code's ``Remote - Containers`` extension is great for this.
Open VS Code and chose to open the directory containing the repository. Once you
do this you may be prompted to open the directory in a container. Select this or
search for ``Remote-Containers: Open Folder in Container...`` in the Command Palette
(ctrl + shift + p or in ``View`` drop down menu). 

You can open a terminal (``Terminal -> New Terminal``) and it will open in your
workspace and execute commands with your container. You can use VS Code's Python
Debugger to run and debug files and this will run within the container. Any changes
you make, however, will be reflected locally. 

If you wish to develop without attaching an editor to the container, you may run::
    make build
    make develop

This will attach you to bash running within the container. 

Style
--------------------------------------

Pull requests will be checked against the linters defined in ``.pre-commit-config.yaml``.
You may check locally before each commit by running ``pre-commit install``.