from setuptools import setup, find_packages

setup(
    name='coalsql',
    version='0.0.0',
    description='COAL-SQL: Coverage-Guided Augmentation and Failure-Driven Learning for Text-to-SQL Post-Training',
    packages=find_packages(include=[]),
    install_requires=[
        'sentence_transformers',
        'tabulate',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
