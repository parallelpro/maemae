from setuptools import setup, find_packages

setup(
    name='maemae',  # Replace with your package name
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    description='A description of your package',
    author='Your Name',
    author_email='your.email@example.com',
    install_requires=['numpy', 'scipy', 'jax', 'jaxlib', 'finufft']  # Add any dependencies if needed
)