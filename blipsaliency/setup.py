from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="blipsaliency",
    version="1.0.1",
    author="MinghaoLiu",
    description="BLIP BeaconSaliency: Harnessing BERT and ViT Synergy for Text-Guided Visual Focus in Deep Learning Models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Text-Guided Visual Saliency, BERT and ViT Synergy, DNN models, Visual Attention Mechanism, Multi-Modal Learning, Grad-CAM integration, Image-Text Alignment, Saliency Map Prediction, pytorch",
    packages=find_namespace_packages(include="blipsaliency.*"),
    python_requires=">=3.7.0",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)
