import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.2.6'
PACKAGE_NAME = 'shaphypetune'
AUTHOR = 'Marco Cerliani'
AUTHOR_EMAIL = 'cerlymarco@gmail.com'
URL = 'https://github.com/cerlymarco/shap-hypetune'

LICENSE = 'MIT'
DESCRIPTION = 'A python package for simultaneous Hyperparameters Tuning and Features Selection for Gradient Boosting Models.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

REQUIREMENTS_FILE = HERE / "requirements.txt"
INSTALL_REQUIRES = [
       line.strip() for line in REQUIREMENTS_FILE.read_text().splitlines()
       if line.strip() and not line.startswith('#')
   ]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3',
      packages=find_packages()
      )