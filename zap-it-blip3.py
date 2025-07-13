"""
zap-it-blip3.py

Holds the Blip3QA class — a lightweight wrapper around the open‑source
BLIP‑3 / XGen‑MM vision‑language model for visual question answering.

Design goals
------------
* Mirror the public interface style of ClipFilter
* Single‑image / single‑question only (no tiling, cropping or batching)
* Pure inference — no fine‑tuning, no captioning, no retrieval
"""

import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)

class _EosListStoppingCriteria(StoppingCriteria):
    """
    Stops generation when the special BLIP‑3 end‑of‑answer sequence appears.
    The official model card specifies [32007] as the default sequence.
    """
    def __init__(self, eos_sequence=(32007,)):
        self.eos_sequence = list(eos_sequence)

    def __call__(self, input_ids, _scores, **kwargs):
        if len(input_ids[0]) < len(self.eos_sequence):
            return False
        return input_ids[0][-len(self.eos_sequence):].tolist() == self.eos_sequence


class Blip3QA:
    """
    Example
    -------
    >>> qa = Blip3QA({})
    >>> img = Image.open("dog.jpg").convert("RGB")
    >>> print(qa.answer(img, "How many dogs are there?"))
    'There are two dogs in the picture.'
    """

    def __init__(self,
                 blip_config: dict,
                 device: str = "cuda",
                 verbosity: int = 1,
                 log_print_func=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        # Allow override of the checkpoint name
        self.model_name = blip_config.get(
            "model_name",
            "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        )

        self.log_print(f"[Blip3QA] loading {self.model_name}", 1, self.verbosity)

        # --- load assets -----------------------------------------------------
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,  # BLIP‑3 tokenizer requires slow tokenizer
            legacy=False
        )
        # add special tokens used by the vision backbone
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # prompt template defined by model authors
        self._prompt = (
            "<|system|>\nA chat between a curious user and an artificial "
            "intelligence assistant. The assistant gives helpful, detailed, "
            "and polite answers to the user's questions.<|end|>\n"
            "<|user|>\n<image>\n{q}<|end|>\n<|assistant|>\n"
        )

        self.stopper = _EosListStoppingCriteria()

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def answer(self, image, query: str, max_new_tokens: int = 768) -> str:
        """
        Args
        ----
        image : PIL.Image.Image   or   np.ndarray (H,W,3, uint8/rgb)
        query : str
        Returns
        -------
        answer : str
        """
        if not isinstance(image, Image.Image):
            # assume NumPy array with RGB ordering
            image = Image.fromarray(image)

        # vision side
        vision_inputs = self.image_processor(
            [image],
            return_tensors="pt",
            image_aspect_ratio="anyres"
        )

        # language side
        prompt = self._prompt.format(q=query)
        lang_inputs = self.tokenizer([prompt], return_tensors="pt")

        # merge & push to device
        inputs = {**vision_inputs, **lang_inputs}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # generate
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                image_size=[image.size],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
                top_p=None,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[self.stopper],
            )

        text = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True
        )
        # the model returns "<|end|>" after the answer — strip everything after it
        return text.split("<|end|>")[0].strip()
