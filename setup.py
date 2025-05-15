from setuptools import setup, find_packages


def read_version():
    version = {}
    with open("src/codemixtoolkit/_version.py") as f:
        exec(f.read(), version)
    return version["__version__"]


setup(
    name="codemixtoolkit",
    version=read_version(),
    author="Prashant Kodali",
    author_email="prashant.kodali@research.iiit.ac.in",
    description="A toolkit for code-mixed language processing",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "ai4bharat_transliteration==1.1.3",
        "alphabet_detector==0.0.7",
        "datasets==3.3.1",
        "htbuilder==0.9.0",
        "indic_nlp_library==0.92",
        "ipython==8.12.3",
        "litellm==1.69.1",
        "numpy==2.2.5",
        "pandas==2.2.3",
        "python-dotenv==1.1.0",
        "PyYAML==6.0.2",
        "sacrebleu==2.5.1",
        "scikit_learn==1.6.1",
        "setuptools==75.8.0",
        "st_annotated_text==4.0.2",
        "stanza==1.10.1",
        "torch==2.4.0",
        "tqdm==4.67.1",
        "transformers==4.49.0",
    ],
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
