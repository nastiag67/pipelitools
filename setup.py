#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas', 'numpy', 'scipy', 'seaborn', 'matplotlib', 'pandas_profiling',
                      'sklearn', 'datetime', 'xlwings']

test_requirements = ['pytest>=3', ]

setup(
    author="Anastasia Glushkova",
    author_email="anastasia.glushkova0@yandex.ru",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
    ],
    description="Tools for data analysis",
    install_requires=requirements,
    license="MIT license",
    long_description=long_description + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='pipelitool',
    name="tools",
    packages=find_packages(include=['tools', 'tools.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url="https://github.com/nastiag67/tools",
    version="1.0.0",
    zip_safe=False,

)

