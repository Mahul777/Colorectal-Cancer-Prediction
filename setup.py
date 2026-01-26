# setup.py -> used to configure and install the Python package
# It also installs all required dependencies

# Import required functions from setuptools
from setuptools import setup, find_packages

# Open requirements.txt file and read dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Call the setup function to configure the package
setup(
    # Name of the Python package
    name="MLOPS-PROJECT-2",

    # Version of the package
    version="0.1",

    # Author name
    author="Sahu",

    # Automatically find all packages (folders with __init__.py)
    packages=find_packages(),

    # Dependencies to install
    install_requires=requirements,
)
