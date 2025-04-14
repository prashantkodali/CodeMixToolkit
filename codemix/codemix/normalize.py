from ai4bharat.transliteration import XlitEngine


class Normalizer:

    def __init__(self, tgt_script="en", src_script="hi", src_script_type="indic"):
        self.tgt_script = tgt_script
        self.src_script = src_script
        self.src_script_type = src_script_type
        self.engine = XlitEngine(beam_width=10, src_script_type=src_script_type)

    def normalize_text(self, text: str) -> str:
        out = self.engine.translit_sentence(text, self.src_script)
        return out


def normalize_text(
    text: str, tgt_script="en", src_script="hi", src_script_type="indic"
) -> str:
    raise NotImplementedError


def romanize_text(
    text: str, tgt_script="en", src_script="hi", src_script_type="indic"
) -> str:
    e = XlitEngine(beam_width=10, src_script_type=src_script_type)
    out = e.translit_sentence(text, src_script)
    return out
