import json
import sys
from typing import Tuple

sys.path.append("../")
from overrides import overrides


class Prompt(object):
    def get_target(self, label: str, num_mask: int, mask_sym: str) -> str:
        return label if num_mask <= 0 else " ".join([mask_sym] * num_mask)

    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        return prompt.replace("[X]", label), label

    def fill_y(
        self,
        prompt: str,
        label: str,
        num_mask: int = 0,
        mask_sym: str = "[MASK]",
    ) -> Tuple[str, str]:
        return prompt.replace("[Y]", self.get_target(label, num_mask, mask_sym)), label

    @staticmethod
    def instantiate(lang: str, *args, **kwargs):
        if lang == "bio_default":
            return PromptBioDefault(*args, **kwargs)
        return Prompt(*args, **kwargs)


class PromptBioDefault(Prompt):
    def __init__(self):
        super().__init__()

    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        return prompt.replace(["[X]"], label), label

    @overrides
    def fill_y(
        self,
        prompt: str,
        label: str,
        num_mask: int = 0,
        mask_sym: str = "[MASK]",
    ) -> Tuple[str, str]:

        target = self.get_target(label, num_mask, mask_sym)

        return prompt.replace(["[Y]"], target), label
