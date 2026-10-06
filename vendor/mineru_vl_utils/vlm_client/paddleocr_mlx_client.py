"""PaddleOCR-VL 1.6 element recognizer for MinerU's native-layout hybrid pipeline.

The layout stage must remain PP-DocLayout. This client never sends MinerU's
Layout Detection task to PaddleOCR-VL, which was not trained for that task.
"""
import re
from .mlx_client import MlxVlmClient
from .base_client import SamplingParams, UnsupportedError


def normalize_math_delimiters(text: str) -> str:
    """Normalize model math markers, protecting existing math, code and prices."""
    protected = re.compile(r"```[\s\S]*?```|`[^`\n]*`|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]")
    dollars = re.compile(r"(?<!\\)\$\$(?P<display>[\s\S]*?)(?<!\\)\$\$|(?<![\\$])\$(?!\$)(?P<inline>[^$\n]+?)(?<!\\)\$(?!\$)")

    def part(chunk):
        def replacement(match):
            display = match.group('display')
            body = display if display is not None else match.group('inline')
            # Nested dollar formats are ambiguous; do not introduce a new layer.
            if '$' in body:
                return match.group(0)
            stripped = body.strip()
            if not stripped:
                return match.group(0)
            if display is not None:
                return r'\[' + body + r'\]'
            # A dollar price remains a price, including paired pure numerals.
            context = chunk[max(0,match.start()-20):match.start()]
            if re.fullmatch(r'[+-]?\d+(?:[.,]\d+)*',stripped):
                return match.group(0)
            if stripped[0].isdigit() and re.search(r'价格|单价|金额|费用|售价|人民币|美元|合计|price|cost|USD',context,re.I):
                return match.group(0)
            command = bool(re.search(r'\\[A-Za-z]+|\\[{}]',body))
            if not command and (stripped[-1] in '+-*/=<>^_' or body[-1].isspace()):
                return match.group(0)
            symbols = bool(re.fullmatch(r'\s*[A-Za-z]+(?:\s*[,;]\s*[A-Za-z]+)*\s*',body))
            function = bool(re.fullmatch(r'\s*[A-Za-z][A-Za-z0-9]*\([^$\n]*\)\s*',body))
            operator = bool(re.search(r'[=^_+*/<>]',body))
            if not (command or symbols or function or operator):
                return match.group(0)
            return r'\(' + body + r'\)'
        return dollars.sub(replacement,chunk)

    pieces=[];last=0
    for match in protected.finditer(text):
        pieces.append(part(text[last:match.start()]));pieces.append(match.group(0));last=match.end()
    pieces.append(part(text[last:]))
    return ''.join(pieces)


class PaddleOCRMlxClient(MlxVlmClient):
    def __init__(self, model, processor, **kwargs):
        if getattr(model.config, 'model_type', '') != 'paddleocr_vl':
            raise ValueError('PaddleOCRMlxClient requires the PaddleOCR-VL model architecture.')
        kwargs['system_prompt'] = ''
        kwargs['allow_truncated_content'] = False
        kwargs['batch_size'] = kwargs.get('batch_size') or 8
        super().__init__(model, processor, **kwargs)
        self.prefill_batch_size = 4

    @staticmethod
    def task_prompt(prompt: str) -> str:
        if 'Layout Detection:' in prompt:
            raise UnsupportedError('PaddleOCR-VL must use PP-DocLayout; the MinerU VLM layout task is unsupported.')
        if 'Formula Recognition:' in prompt:
            return 'Formula Recognition:'
        if 'Table Recognition:' in prompt:
            return 'Table Recognition:'
        if 'Chart Recognition:' in prompt:
            return 'Chart Recognition:'
        # Preserve the complete Chinese + inline LaTeX sequence in each ROI.
        return 'OCR:'

    def build_messages(self, prompt: str, has_image: bool = True) -> list[dict]:
        return super().build_messages(self.task_prompt(prompt), has_image=has_image)

    def _batch_kwargs(self, params, count):
        kwargs = super()._batch_kwargs(params, count)
        # Smaller prefill dispatches keep the macOS GPU watchdog responsive;
        # eight completion slots and the full token budget remain available.
        kwargs['prefill_step_size'] = 512
        return kwargs

    def _predict_one(self, image, prompt, sampling_params):
        text = super()._predict_one(image, prompt, sampling_params)
        return normalize_math_delimiters(text) if self.task_prompt(prompt) == 'OCR:' else text

    def _predict_batch(self, images, prompts, params):
        outputs = super()._predict_batch(images, prompts, params)
        return [normalize_math_delimiters(text) if self.task_prompt(prompt) == 'OCR:' else text
                for text, prompt in zip(outputs, prompts)]

    def build_sampling_params(self, sampling_params: SamplingParams | None) -> SamplingParams:
        # MinerU penalties are tuned for a different model. Greedy OCR should
        # allow repeated Chinese phrases, mathematical symbols and table cells.
        requested = sampling_params.max_new_tokens if sampling_params is not None else None
        if requested is None and self.sampling_params is not None:
            requested = self.sampling_params.max_new_tokens
        return SamplingParams(temperature=0.0, top_p=1.0, top_k=0,
                              presence_penalty=0.0, frequency_penalty=0.0,
                              repetition_penalty=None, no_repeat_ngram_size=None,
                              max_new_tokens=requested or 4096)


__all__ = ['PaddleOCRMlxClient']
