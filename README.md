Integrate Model

Place weights in ./weights/ (e.g. ./weights/best.pt, ./weights/coralscope.pth).

Edit inference.py:

In CoralPipeline.__init__: load your detector/segmenter and move to self.device.

In CoralPipeline.run(img): perform detection + segmentation, create an overlay image, and return:

overlay — a PIL.Image with masks/boxes drawn

results — a dict with meta/metrics (shown on the right panel)


Run — Option A (No Docker, quickest)

Prerequisite: Python 3.10+

base:
# create & activate venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# install deps
pip install -r requirements.txt

# make sure your weights are in ./weights/ and inference.py loads them
streamlit run app.py

# open in your browser:
# Open http://localhost:8501
 (do not use 0.0.0.0 in the browser).