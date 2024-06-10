from setuptools import setup, find_packages

setup(
    name='shrimpgrad',
    author='Kevin Kenyon',
    license='MIT',
    version='0.0.1',
    python_requires='>=3.11',
    packages=find_packages(),
    install_requires=["graphviz", 'numpy', 'cffi' ],
    extras_require={
      'testing': [
          "torch",
          "pytest",
          "networkx",
      ],
    },
)
