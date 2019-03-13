from setuptools import setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='trans_mri',
      version='0.1',
      description='Fast transfer learning experimentation with MRI data in PyTorch',
      url='http://github.com/ben0it8/trans_mri',
      author='Oliver Atanaszov',
      author_email='oliver.atanaszov@gmail.com',
      license='MIT',
      packages=['trans_mri'],
      install_requires=required,
      zip_safe=False)
