from setuptools import setup
Version="1.1.2"
setup (name = "durolib",
       author="Paul J. Durack (durack1@llnl.gov)",
       version=Version,
       description = "Python utilities for climate",
       url = "http://github.com/durack1/durolib",
       packages = ['durolib'],
       package_dir = {'durolib': 'lib'},
       data_files = [('durolib/data',['data/CMIP5BranchTimes.json',
                              'data/CMIP5BranchTimes.pickle'])],
      )