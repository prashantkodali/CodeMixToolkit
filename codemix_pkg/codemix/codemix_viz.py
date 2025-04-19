# import streamlit as st
# from annotated_text import annotated_text
# from st-annotated-text import annotation
import streamlit as st
from annotated_text import annotated_text, annotation
import html
from htbuilder import H, HtmlElement, styles
from htbuilder.units import unit
from ast import literal_eval
from IPython.display import HTML, display

# Only works in 3.7+: from htbuilder import div, span
div = H.div
span = H.span

# Only works in 3.7+: from htbuilder.units import px, rem, em
px = unit.px
rem = unit.rem
em = unit.em


class AnnotatedTextPrinter:
    def __init__(self):
        self.lang_color_dict = {
            "en": "#afa",
            "hi": "#faa",
            "te": "#f0f",
            "ta": "#0ff",
            "gu": "#ff0",
            "ka": "#0f0",
            "ml": "#f00",
            "ne": "#8ef",
            "acro": "#fea",
            "univ": "#c39",
        }

    def print_sample_st_annot_text(self, sample_text, sample_langspan, sample_posspan):
        if not isinstance(sample_text, list):
            sample_text = sample_text.split()

        assert len(sample_text) == len(sample_posspan) == len(sample_langspan)

        annot_text = []

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
                raise Exception("Oh noes!")

        # display(HTML('<hr>'))
        # display(HTML(str(out)))
        # display(HTML('<hr>'))

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
            {str(out)}
        </body>
        </html>
        """

        # Save to file
        with open("codemix_visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
            print("HTML file saved as codemix_visualization.html")

        # Display in notebook (if running in notebook)
        display(HTML("<hr>"))
        display(HTML(str(out)))
        display(HTML("<hr>"))
