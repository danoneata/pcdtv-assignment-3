import json

from pathlib import Path

import click
import librosa
import torch

from omegaconf import OmegaConf
from sklearn.metrics import roc_curve
from tqdm import tqdm

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from speechbrain.inference.speaker import EncoderClassifier

from train import load_config


class SpeakerVerificationScorer:
    def score(self, audio1, audio2) -> float:
        raise NotImplementedError


class ECAPATDNNScorer(SpeakerVerificationScorer):
    def __init__(self, *args, **kwargs):
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"},
        )
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def score(self, audio1, audio2) -> float:
        with torch.no_grad():
            sim = self.similarity(
                self.classifier.encode_batch(audio1),
                self.classifier.encode_batch(audio2),
            )
        return sim.cpu().item()


class OurScorer(SpeakerVerificationScorer):
    def __init__(self, config_name: str):
        raise NotImplementedError

    def score(self, audio1, audio2) -> float:
        raise NotImplementedError


def compute_eer(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return 100 * eer


SCORERS = {
    "ecapa-tdnn": ECAPATDNNScorer,
    "ours": OurScorer,
}


def evaluate(scorer_name, config_name):
    with open("data/voxceleb1/a3/test-pairs.json") as f:
        pairs = json.load(f)

    system = SCORERS[scorer_name](config_name)
    config = load_config(config_name)

    SAMPLING_RATE = 16_000

    def load_audio(subpath):
        path = Path(config.data_dir) / "wav" / subpath
        audio, _ = librosa.load(path, sr=SAMPLING_RATE)
        return torch.tensor(audio).unsqueeze(0)

    labels = [pair["label"] for pair in pairs]
    scores = [
        system.score(load_audio(pair["path1"]), load_audio(pair["path2"]))
        for pair in tqdm(pairs)
    ]

    print(compute_eer(labels, scores))


@click.command()
@click.option(
    "-s",
    "--scorer",
    "scorer_name",
    type=str,
    required=True,
)
@click.option(
    "-c",
    "--config",
    "config_name",
    type=str,
    required=True,
)
def main(scorer_name, config_name):
    evaluate(scorer_name, config_name)


if __name__ == "__main__":
    main()
