#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from pathlib import Path
from setuptools import setup

root_path = Path(__file__).resolve().parent


def parse_environment_yml():
    """Parse the environment.yml file and extract the package names."""
    # here we cannot use pyyaml because it is not installed yet
    with open(root_path / "environment.yml") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    dependencies = []
    inside_dependencies = False
    for entry in lines:
        if entry == "dependencies:":
            inside_dependencies = True
            continue
        if inside_dependencies:
            if not entry.startswith("-"):
                break
            dependency_name = entry.strip().split(" ")[-1]
            if dependency_name != "pip:" and "python=" not in dependency_name:
                dependencies.append(dependency_name)

    sanitized_dependencies = []
    for dependency in dependencies:
        # conda package ignite is named pytorch-ignite on pypi
        if "ignite" in dependency:
            dependency = "pytorch-" + dependency
        if dependency.startswith("pytorch="):
            dependency = dependency.replace("pytorch", "torch")
        if "=" in dependency and "==" not in dependency:
            dependency = "==".join(dependency.split("="))
        if "http://" in dependency or "https://" in dependency:
            package_name = dependency.split("/")[-1].split(".")[0]
            dependency = package_name + "@" + dependency
        sanitized_dependencies.append(dependency)
    return sanitized_dependencies


this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text(encoding="utf-8")

setup_requirements = ["pytest-runner"]

setup(
    author="Robin van de Water",
    author_email="robin.vandewater@hpi.de",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    description="A modular preprocessing package for a Pandas Dataframe.",
    # install_requires=parse_environment_yml(),
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="recipies",
    name="recipies",
    packages=["recipys"],
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=[],
    url="https://github.com/rvandewater/recipys",
    version="0.1.3",
    zip_safe=False,
)
