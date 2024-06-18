{ pkgs ? import <nixpkgs> {} }:
with pkgs.python3Packages;

pkgs.mkShell {
  buildInputs = [
    (buildPythonPackage rec {
      name = "shrimpgrad";
      src = ./.;
      propagatedBuildInputs = [ scikit-learn pylint pytest numpy cffi graphviz torch networkx pydot];
    })
  ];
}
