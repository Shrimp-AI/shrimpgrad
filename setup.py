from setuptools import setup

setup(
    name='shrimpgrad',
    author='Kevin Kenyon',
    license='MIT',
    version='0.0.1',
    python_requires='>=3.8',
    packages=['shrimpgrad', 'shrimpgrad.scalar'],
    package_data={'shrimpgrad': ['lib/*.dylib']},
    install_requires=["graphviz", 'numpy', 'cffi', 'llvmlite'],
    extras_require={
      'testing': [
          "torch",
          "pytest",
      ],
    },
)
