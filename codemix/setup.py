from setuptools import setup

setup(
    name="codemix",
    version="1.0",
    author="Prashant Kodali",
    author_email="prashant.kodali@research.iiit.ac.in",
    description="A toolkit for code-mixed language processing",
    packages=["codemix"],
    install_requires=[
        "statistics",
        "streamlit",
        "htbuilder",
        "IPython",
        "st-annotated-text",
        "pandas",
        "stanza",
        "indic-nlp-library",
        "torch==2.4.0",
        "transformers",
        "datasets",
        "alphabet-detector",
        "ai4bharat-transliteration",
        "python-dotenv",
    ],
)
