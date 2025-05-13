from ai4bharat.transliteration import XlitEngine
from typing import Literal


class Normalizer:
    """
    A class for normalizing text using AI4Bharat's transliteration engine.

    This class provides functionality to transliterate text from one script to another,
    primarily focused on Indic scripts to English and vice versa.

    Attributes:
        tgt_script (str): Target script for transliteration (default: "en")
        src_script (str): Source script for transliteration (default: "hi")
        src_script_type (str): Type of source script (default: "indic")
        engine (XlitEngine): Instance of AI4Bharat's transliteration engine
    """

    def __init__(
        self,
        tgt_script: str = "en",
        src_script: str = "hi",
        src_script_type: Literal["indic", "roman"] = "indic",
    ) -> None:
        """
        Initialize the Normalizer with specified script parameters.

        Args:
            tgt_script (str): Target script for transliteration (default: "en")
            src_script (str): Source script for transliteration (default: "hi")
            src_script_type (Literal["indic", "roman"]): Type of source script (default: "indic")
        """
        self.tgt_script = tgt_script
        self.src_script = src_script
        self.src_script_type = src_script_type
        self.engine = XlitEngine(beam_width=10, src_script_type=src_script_type)

    def normalize_text(self, text: str) -> str:
        """
        Normalize the input text by transliterating it to the target script.

        Args:
            text (str): Input text to be normalized

        Returns:
            str: Normalized text in the target script
        """
        out = self.engine.translit_sentence(text, self.src_script)
        return out


def normalize_text(
    text: str,
    tgt_script: str = "en",
    src_script: str = "hi",
    src_script_type: Literal["indic", "roman"] = "indic",
) -> str:
    """
    Normalize text using the AI4Bharat transliteration engine.

    Args:
        text (str): Input text to be normalized
        tgt_script (str): Target script for transliteration (default: "en")
        src_script (str): Source script for transliteration (default: "hi")
        src_script_type (Literal["indic", "roman"]): Type of source script (default: "indic")

    Returns:
        str: Normalized text in the target script

    Raises:
        NotImplementedError: This function is not yet implemented
    """
    raise NotImplementedError


def romanize_text(
    text: str,
    tgt_script: str = "en",
    src_script: str = "hi",
    src_script_type: Literal["indic", "roman"] = "indic",
) -> str:
    """
    Convert text to Roman script using AI4Bharat's transliteration engine.

    Args:
        text (str): Input text to be romanized
        tgt_script (str): Target script for transliteration (default: "en")
        src_script (str): Source script for transliteration (default: "hi")
        src_script_type (Literal["indic", "roman"]): Type of source script (default: "indic")

    Returns:
        str: Romanized text
    """
    e = XlitEngine(beam_width=10, src_script_type=src_script_type)
    out = e.translit_sentence(text, src_script)
    return out
