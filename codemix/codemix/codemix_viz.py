"""
Module for visualizing code-mixed text with language and POS annotations.

This module provides functionality to create and display annotated text where each token
is color-coded based on its language and includes POS information. It supports both
Jupyter notebook display and HTML export capabilities.
"""

# import streamlit as st
# from annotated_text import annotated_text
# from st-annotated-text import annotation
from annotated_text import annotation
import html
from htbuilder import HtmlElement
from htbuilder.units import unit
from IPython.display import HTML, display
from typing import Optional, List, Tuple, Dict, Union

# Only works in 3.7+: from htbuilder import div, span
# div = H.div
from htbuilder import div

# span = H.span

# Only works in 3.7+: from htbuilder.units import px, rem, em
px = unit.px
rem = unit.rem
em = unit.em


class AnnotatedTextPrinter:
    """
    A class for creating and displaying annotated text with language and POS information.
    
    This class provides methods to create color-coded text annotations where each token
    is associated with a language and POS tag. It supports both interactive display in
    Jupyter notebooks and HTML export for static visualization.
    """
    
    def __init__(self) -> None:
        """
        Initialize the AnnotatedTextPrinter with a predefined color scheme for different languages.
        """
        # self.lang_color_dict = {
        #     "en": "#afa",
        #     "hi": "#faa",
        #     "te": "#f0f",
        #     "ta": "#0ff",
        #     "gu": "#ff0",
        #     "ka": "#0f0",
        #     "ml": "#f00",
        #     "ne": "#8ef",
        #     "acro": "#fea",
        #     "univ": "#c39"}
        
        # self.lang_color_dict = {
        #     "en": "#0072B2",  # Blue
        #     "hi": "#D55E00",  # Orange
        #     "te": "#009E73",  # Green
        #     "ta": "#F0E442",  # Yellow
        #     "gu": "#CC79A7",  # Pink
        #     "ka": "#56B4E9",  # Light Blue
        #     "ml": "#E69F00",  # Orange-Yellow
        #     "ne": "#000000",  # Black
        #     "acro": "#999999", # Gray
        #     "univ": "#FFFFFF"  # White
        # }
        
        self.lang_color_dict: Dict[str, str] = {
                                "en": "#1f77b4",  # Blue
                                "hi": "#D55E00",  # Orange
                                "te": "#2ca02c",  # Green
                                "ta": "#d62728",  # Red
                                "gu": "#9467bd",  # Purple
                                "ka": "#8c564b",  # Brown
                                "ml": "#e377c2",  # Pink
                                "ne": "#7f7f7f",  # Gray
                                "acro": "#bcbd22", # Yellow-Green
                                "univ": "#17becf"  # Cyan
                                }
        
        self.str_html: Optional[str] = None

    def print_sample_st_annot_text(
        self, 
        sample_text: Union[str, List[str]], 
        sample_langspan: List[str], 
        sample_posspan: List[str]
    ) -> None:
        """
        Create and display annotated text with language and POS information.
        
        Args:
            sample_text: Input text as either a string (will be split) or list of tokens
            sample_langspan: List of language codes corresponding to each token
            sample_posspan: List of POS tags corresponding to each token
            
        Raises:
            AssertionError: If the lengths of sample_text, sample_langspan, and sample_posspan don't match
            Exception: If an invalid annotation type is encountered
        """
        if not isinstance(sample_text, list):
            sample_text = sample_text.split()

        assert len(sample_text) == len(sample_posspan) == len(sample_langspan)

        annot_text: List[Tuple[str, str, str]] = []

        for form, lang, pos in zip(sample_text, sample_langspan, sample_posspan):
            annot_text.append((form, pos, self.lang_color_dict[lang]))

        out = div()

        for arg in annot_text:
            if isinstance(arg, str):
                out(html.escape(arg))
            elif isinstance(arg, HtmlElement):
                out(arg)
            elif isinstance(arg, tuple):
                out(annotation(*arg))
            else:
                raise Exception("Invalid annotation type encountered")

        # display(HTML('<hr>'))
        # display(HTML(str(out)))
        # display(HTML('<hr>'))
        
        self.str_html = str(out)
        
        # Display in notebook (if running in notebook)
        display(HTML('<hr>'))
        display(HTML(str(out)))
        display(HTML('<hr>'))
        
    def export_html(self, file_name: Optional[str] = None) -> None:
        """
        Export the annotated text visualization to an HTML file.
        
        Args:
            file_name: Optional name for the output HTML file. If not provided,
                      defaults to "codemix_visualization.html"
        """
        # Create the HTML string
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CodeMix Visualization</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    padding: 20px;
                }}
                .token {{
                    display: inline-flex;
                    flex-direction: row;
                    align-items: center;
                    border-radius: 0.5rem;
                    padding: 0.25rem 0.5rem;
                    margin: 0.1rem;
                }}
                .token-info {{
                    font-size: 0.75rem;
                    opacity: 0.5;
                    margin-left: 0.5rem;
                }}
            </style>
        </head>
        <body>
            {self.str_html}
        </body>
        </html>
        """

        # Save to file
        if file_name is None:
            file_name = "codemix_visualization.html"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(html_content)
            print(f"HTML file saved as {file_name}")

