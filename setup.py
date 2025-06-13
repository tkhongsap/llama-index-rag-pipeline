from setuptools import setup, find_packages

setup(
    name="src_iLand",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.0",
        "psycopg2-binary>=2.9.0",
        "tqdm>=4.62.0",
        "python-dateutil>=2.8.2",
        "langdetect>=1.0.9",
        "chardet>=4.0.0",
        "llama-index-core>=0.9.0",
    ],
    python_requires=">=3.8",
) 