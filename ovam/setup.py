from setuptools import setup, find_packages

setup(
    name='ovam',
    version='0.0.1',
    description='A brief description of my package',
    long_description='''\
        A longer description of my package.
        You can use Markdown or reStructuredText syntax here.
    ''',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/my_package',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # Add any dependencies your package requires
        'numpy',
        'matplotlib',
    ],
)