from setuptools import setup, find_packages

setup(
    name="your-project-name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'pandas',
        'Pillow',
        'tqdm',
        'seaborn',
        'scikit-learn'
    ],
    python_requires='>=3.7'
)