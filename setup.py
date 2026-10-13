
#f i want to use src as my local file i want to use this.
from setuptools import setup, find_packages

setup(
    name = "generative ai project",
    version = "0.1",
    packages = find_packages(), # it will check where __init__.py file is present and it will consider that folder as local package.
    install_requires = [],
    author = "Pakanti Teja",
    author_email = "tejapakanati1@gamail.com",
)