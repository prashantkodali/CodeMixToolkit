from setuptools import setup, find_packages


def read_version():
    version = {}
    with open("src/codemixtoolkit/_version.py") as f:
        exec(f.read(), version)
    return version["__version__"]


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="codemixtoolkit",
    version=read_version(),
    author="Prashant Kodali",
    author_email="prashant.kodali@research.iiit.ac.in",
    description="A toolkit for code-mixed language processing",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
)

# [
#     "statistics",
#     "streamlit",
#     "htbuilder",
#     "IPython",
#     "st-annotated-text",
#     "pandas",
#     "stanza",
#     "indic-nlp-library",
#     "torch==2.4.0",
#     "transformers",
#     "datasets",
#     "alphabet-detector",
#     "ai4bharat-transliteration",
#     "python-dotenv",
#     "litellm",
# ],
