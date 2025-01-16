# Find packages within libaries
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) ->List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        lines = file_obj.readlines()
        # requirements = [req.replace("\n", "") for req in requirements]
        for req in lines:
            requirements.append(req.replace("\n", ""))   # Strip newline characters and process each requirement
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name="student_performance_project",
    version="0.0.1",
    author="danny",
    packages=find_packages(),
    install_requires= get_requirements("requirements.txt")
    )