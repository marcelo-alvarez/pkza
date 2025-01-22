from setuptools import setup
pname='pkza'
setup(name=pname,
      version='0.1',
      description='Zeldovich power spectrum',
      url='http://github.com/marcelo-alvarez/pkza',
      author='Marcelo Alvarez',
      license='MIT',
      packages=['pkza'],
      package_dir={'pkza': 'pkza'},
      package_data={
        'pkza': ["data/*"]
      },
      zip_safe=False)
