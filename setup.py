from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='forcedirected',
    version='0.2.2',
    author='Hamidreza Lotfalizadeh (Hessam)',
    author_email='hlotfali_at_purdue_edu',
    description='Force-Directed graph embedding',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HessamLa/forcedirected',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'forcedirected=forcedirected.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',
    install_requires=[
        'click>=8.1.3',
        'matplotlib>=3.8.3',
        'networkit>=10.1',
        'networkx>=3.0',
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'scikit_learn>=1.2.2',
        'setuptools>=67.6.1',
        'torch>=2.0.0',
        'torch_geometric>=2.3.0',
        'recursivenamespace @ git+https://github.com/HessamLa/RecursiveNamespace.git#egg=recursivenamespace',
    ],
)



