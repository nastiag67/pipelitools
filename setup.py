from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tools",
    version="1.0.0",
    author="Anastasia Glushkova",
    author_email="anastasia.glushkova0@yandex.ru",
    description="Tools for data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nastiag67/tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
