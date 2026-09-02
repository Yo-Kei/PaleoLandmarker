from paleolandmarker.baselines.deeplabcut_style import DeepLabCutImageNetBaseline, DeepLabCutStyleBaseline, DeepLabCutSyntheticBaseline
from paleolandmarker.baselines.landmarker_style import LandmarkerStyleBaseline, LandmarkerSyntheticBaseline
from paleolandmarker.baselines.ml_morph_style import MLMorphStyleBaseline
from paleolandmarker.baselines.vandaele2018_style import Vandaele2018StyleBaseline


CANONICAL_BASELINE_REGISTRY = {
    "deeplabcut_imagenet": DeepLabCutImageNetBaseline,
    "deeplabcut_scratch": DeepLabCutStyleBaseline,
    "deeplabcut_synthetic": DeepLabCutSyntheticBaseline,
    "landmarker_scratch": LandmarkerStyleBaseline,
    "landmarker_synthetic": LandmarkerSyntheticBaseline,
    "ml_morph": MLMorphStyleBaseline,
    "vandaele2018": Vandaele2018StyleBaseline,
}

# Former public names remain accepted as CLI aliases. New results and output
# directories always use the canonical names above.
BASELINE_ALIASES = {
    "deeplabcut_style": "deeplabcut_scratch",
    "landmarker_style": "landmarker_scratch",
    "ml_morph_style": "ml_morph",
    "vandaele2018_style": "vandaele2018",
}


def canonical_baseline_name(name):
    return BASELINE_ALIASES.get(name, name)


BASELINE_REGISTRY = CANONICAL_BASELINE_REGISTRY
