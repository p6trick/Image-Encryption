# Third-party data and models

CLIPSeg checkpoint: CIDAS/clipseg-rd64-refined, revision 999e0328d9e10b484360c477313983f9afdd7050, model card declares Apache-2.0. See CLIPSeg_MODEL_CARD.md and TRANSFORMERS_LICENSE.txt (Apache-2.0 text).

Transformers is installed separately; its license is included for attribution. The project imports its public CLIPSeg implementation.

The six demo images come from scikit-image; see SCIKIT_IMAGE_LICENSE.txt and the upstream data documentation for individual image sources: https://scikit-image.org/docs/stable/api/skimage.data.html .

RefCOCO annotations are from jxu124/refcoco; repository revision and annotation checksum are in benchmarks/refcoco/provenance.json. Original COCO image metadata, including image license IDs and source URLs, is retained in manifest_20_seed42.json. Individual image rights remain with their original owners. The sample is included for coursework evaluation.

pycocotools mask conversion is imported as a dependency; see PYCOCOTOOLS_LICENSE.txt. No external model is trained from scratch here. The learned region-transform matrix is independent of CLIPSeg.
