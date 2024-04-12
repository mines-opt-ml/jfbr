from setuptools import setup, find_packages
import os

# Read requirements from 'requirements.txt'
with open('requirements.txt') as f: 
    required = f.read().splitlines() 

setup(
    name='single_loop_deq_simplified',  # You can give any name to your package
    version='0.1',  # Version number for your package
    packages=find_packages(),  # Automatically find packages in the folder
    install_requires=required,  # List of dependencies
    include_package_data=True,  # Include other files like *.txt, *.md etc.
)