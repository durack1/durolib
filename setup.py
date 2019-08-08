from setuptools import setup

# Set package version
Version='1.1.2'

# Call setup function - see https://setuptools.readthedocs.io/en/latest/setuptools.html#adding-setup-arguments
setup(
      name = 'durolib',
      author = 'Paul J. Durack',
      author_email = 'durack1@llnl.gov',
      data_files = [('share/durolib/data',['data/CMIP5BranchTimes.json',
                                           'data/CMIP5BranchTimes.pickle'])],
      description = 'Python utilities for climate',
      packages = ['durolib'],
      url = 'http://github.com/durack1/durolib',
      version = Version,
      )

#      package_dir = {'durolib': 'lib'},
#      package_data = {'durolib': ['data/CMIP5BranchTimes.json',
#                                  'data/CMIP5BranchTimes.pickle']},
#      include_package_data = True,