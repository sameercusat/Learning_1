from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT='-e .'

def get_requiremets(file_path:str)->List[str]:
    ''' THIS  FUNCTION WILL RETURN LIST OF REQUIREMNTS'''
    with open(file_path,'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[x.replace('\n','') for x in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='Learning1',
    version='0.0.1',
    author='Sameer',
    author_email='sameer.cusat2019@gmail.com',
    packages=find_packages(),
    install_requires=get_requiremets('requirements.txt')
)