#!nova: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
from distutils.core import setup
import pkg_resources

pkg_resources.require(["pip >= 19.3.1"])

setup(
    name="kospeech_nova",
    version="latest",
    install_requires=[
        # 'torch==1.7.0',
        "librosa >= 0.7.0",
        "numpy",
        "pandas",
        "tqdm",
        "matplotlib",
        "astropy",
        "sentencepiece",
        "torchaudio==0.6.0",
        "pydub",
        "glob2",
        "omegaconf",
        "datasets",
        "transformers",
        "jiwer == 2.0.0",
    ],
)
