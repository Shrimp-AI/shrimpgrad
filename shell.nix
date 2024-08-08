{ pkgs ? import <nixpkgs> {} }:

with pkgs.python310Packages;

pkgs.mkShell {
  buildInputs = [
    jupyterlab
    ipykernel
    matplotlib
    pip
    virtualenv
    scikit-learn
    pylint
    pytest
    numpy
    cffi
    graphviz
    torch
    networkx
    pydot
  ];

  shellHook = ''
    VENV_DIR=.venv
    KERNEL_NAME="shrimpgrad-kernel"
    KERNEL_DISPLAY_NAME="ShrimpGrad Kernel"

    if [ ! -d "$VENV_DIR" ]; then
      python3.10 -m venv $VENV_DIR
    fi
    source $VENV_DIR/bin/activate

    export PYTHONPATH=$PYTHONPATH:/$(pwd)

    # Ensure pip is using the correct Python version
    #pip install --upgrade pip setuptools wheel

    # Install the package
    pip install --upgrade --no-deps --force-reinstall -e .

    # Create a new Jupyter kernel
    python -m ipykernel install --user --name=$KERNEL_NAME --display-name="$KERNEL_DISPLAY_NAME"

    echo "Virtual environment activated. Package installed in editable mode."
    echo "Jupyter kernel '$KERNEL_DISPLAY_NAME' created."
    echo "Python Path: '$PYTHONPATH'"
  '';
}
