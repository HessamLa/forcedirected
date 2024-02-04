from setuptools import setup, find_packages

setup(
    name='forcedirected',
    version='0.1.0',  # Update the version number as needed
    author='Your Name',
    author_email='your.email@example.com',
    description='Force-Directed graph embedding',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/HessamLa/forcedirected',  # Update with your URL
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'forcedirected=forcedirected.__main__:main',
        ],
    },
    classifiers=[
        # Intended audience, project maturity, license, etc.
        # See: https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Update with your license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',
)