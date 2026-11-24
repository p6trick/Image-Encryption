"""Fixed 20-image RefCOCO validation pilot, using original referring expressions.

python -s evaluate_refcoco.py
No threshold tuning and no sample replacement based on predictions.
"""
import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from pycocotools import mask as mask_api
import requests

from image_protection import configure_cpu, train_key
from text_protection import ROOT, MODEL_REVISION, TextSegmenter, run_image

REPOSITORY = 'jxu124/refcoco'
ANNOTATION = 'data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet'
# One-word prompts for categories present in this fixed 20-image pilot.
# COCO category 88 is teddy bear; use the one-word hypernym 'bear'.
CATEGORY_WORDS = {1: 'person', 21: 'cow', 22: 'elephant', 52: 'banana',
                  56: 'broccoli', 84: 'book', 88: 'bear'}


def metrics(prediction, target):
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    denominator = int(prediction.sum()+target.sum())
    return {'intersection': intersection, 'union': union,
            'iou': intersection/union if union else 1.0,
            'dice': 2*intersection/denominator if denominator else 1.0}


def prepare(count, seed, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    provenance_file = data_dir / 'provenance.json'
    if provenance_file.exists():
        provenance = json.loads(provenance_file.read_text())
    else:
        response = requests.get(f'https://huggingface.co/api/datasets/{REPOSITORY}', timeout=40)
        response.raise_for_status()
        provenance = {'repository': REPOSITORY, 'revision': response.json()['sha'],
                      'official_api': 'https://github.com/lichengunc/refer',
                      'note': 'Third-party mirror; val/train/testA-equivalent/testB counts match UNC release. No independent full-release equivalence audit.'}
    annotation_path = data_dir / Path(ANNOTATION).name
    if not annotation_path.exists():
        response = requests.get(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{provenance["revision"]}/{ANNOTATION}',timeout=90)
        response.raise_for_status()
        annotation_path.write_bytes(response.content)
    actual_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    if provenance.get('annotation_sha256') and provenance['annotation_sha256'] != actual_hash:
        raise ValueError('Annotation checksum mismatch; refusing a different dataset')
    provenance['annotation_sha256'] = actual_hash
    provenance_file.write_text(json.dumps(provenance, indent=2))
    rows = pq.read_table(annotation_path).to_pylist()
    assert len(rows) == 3811 and all(row['split'] == 'val' for row in rows)
    rows.sort(key=lambda row: row['ref_id'])
    random.Random(seed).shuffle(rows)
    selected, seen = [], set()
    for row in rows:
        if row['image_id'] in seen:
            continue
        seen.add(row['image_id'])
        sentence = min(row['sentences'], key=lambda s: s['sent_id'])
        selected.append({'image_id': row['image_id'], 'ref_id': row['ref_id'],
                         'ann_id': row['ann_id'], 'category_id': row['category_id'],
                         'sent_id': sentence['sent_id'], 'text': sentence['raw'],
                         'annotation': json.loads(row['raw_anns']),
                         'image_info': json.loads(row['raw_image_info'])})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Not enough unique images')
    manifest = {'dataset': 'RefCOCO', 'split': 'validation', 'split_convention': 'UNC-compatible mirror',
                'seed': seed, 'count': count, 'selection': 'shuffle refs sorted by ref_id; first ref per unique image; lowest sent_id raw expression',
                'provenance': provenance, 'samples': selected}
    (data_dir / f'manifest_{count}_seed{seed}.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def download_sample(sample, data_dir):
    directory = data_dir / str(sample['ref_id'])
    directory.mkdir(exist_ok=True)
    info = sample['image_info']
    image_path = directory / 'image.jpg'
    if not image_path.exists():
        urls = [f'https://s3.amazonaws.com/images.cocodataset.org/train2014/{info["file_name"]}',
                f'https://images.cocodataset.org/train2014/{info["file_name"]}',
                info.get('flickr_url', '').replace('http://', 'https://')]
        errors = []
        for url in urls:
            if not url:
                continue
            try:
                response = requests.get(url, timeout=40)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                assert image.size == (info['width'], info['height']), 'Image size mismatch'
                image.verify()
                image_path.write_bytes(response.content)
                (directory / 'image_source.txt').write_text(url)
                break
            except Exception as error:
                errors.append(str(error))
        else:
            raise RuntimeError(f'Image {sample["image_id"]} unavailable; sample NOT replaced: {errors}')
    with Image.open(image_path) as image:
        assert image.size == (info['width'], info['height'])
    segmentation = sample['annotation']['segmentation']
    if isinstance(segmentation, list):
        encoded = mask_api.merge(mask_api.frPyObjects(segmentation, info['height'], info['width']))
    elif isinstance(segmentation['counts'], list):
        encoded = mask_api.frPyObjects(segmentation, info['height'], info['width'])
    else:
        encoded = dict(segmentation)
        if isinstance(encoded['counts'], str):
            encoded['counts'] = encoded['counts'].encode()
    target = mask_api.decode(encoded).astype(bool)
    assert target.any() and target.shape == (info['height'], info['width'])
    Image.fromarray(target.astype(np.uint8)*255).save(directory / 'ground_truth.png')
    return directory


def gallery(rows, destination):
    import textwrap
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3.1*len(rows)), squeeze=False)
    for index, (row, axis_row) in enumerate(zip(rows, axes)):
        directory = Path(row['directory'])
        if not directory.is_absolute():
            directory = ROOT / directory
        for ax, name, title in zip(axis_row,
                ['original.png','ground_truth.png','mask.png','protected.preview.png','restored.png'],
                ['Original + text','GT mask','Predicted mask','Protected preview','Restored']):
            skipped = name in ['protected.preview.png','restored.png'] and not row['restoration_verified']
            if skipped:
                ax.text(.5,.5,'Skipped\n(empty prediction)',ha='center',va='center',transform=ax.transAxes,fontsize=10)
            else:
                ax.imshow(Image.open(directory/name), cmap='gray',vmin=0,vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if index == 0: ax.set_title(title,fontsize=10)
        axis_row[0].set_xlabel(textwrap.fill(row['text'],40),fontsize=8)
        axis_row[2].set_xlabel(f'IoU {row["iou"]:.3f} | Dice {row["dice"]:.3f}',fontsize=9)
        if row['restoration_verified']:
            axis_row[-1].set_xlabel(f'Pixel error: {row["max_pixel_error"]}',fontsize=9)
    fig.suptitle('RefCOCO validation subset | CLIPSeg | threshold 0.5',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(destination,dpi=130,facecolor='white')
    plt.close(fig)


def render_galleries(rows, output):
    """Display-only exclusions preserve the original page grouping and metrics."""
    output = Path(output)
    selection_path = output / 'gallery_selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    excluded = set(selection.get('excluded_ref_ids', []))
    for start in range(0, len(rows), 5):
        visible = [row for row in rows[start:start+5] if row['ref_id'] not in excluded]
        destination = output / f'gallery_{start//5+1}.png'
        if visible:
            gallery(visible, destination)
        else:
            destination.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--prompt-mode',choices=['original','word'],default='original')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--curated-gallery',action='store_true',
                        help='Apply presentation-only row exclusions; all samples remain in metrics')
    args=parser.parse_args()
    if args.count < 1: parser.error('count must be positive')
    configure_cpu(4)
    default_output=ROOT/'results'/('refcoco20_words' if args.prompt_mode=='word' else 'refcoco20')
    output=(args.output_dir or default_output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if args.curated_gallery:
        config = ROOT / 'config' / 'gallery_selection.json'
        (output/'gallery_selection.json').write_bytes(config.read_bytes())
    data_dir=ROOT/'benchmarks'/'refcoco'
    manifest=prepare(args.count,args.seed,data_dir)
    manifest['prompt_mode']=args.prompt_mode
    for sample in manifest['samples']:
        sample['original_text']=sample['text']
        if args.prompt_mode=='word':
            if sample['category_id'] not in CATEGORY_WORDS:
                raise ValueError(f'Add a one-word mapping for COCO category {sample["category_id"]}')
            sample['text']=CATEGORY_WORDS[sample['category_id']]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Fixed {args.count} unique images; output: {output}',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        directories=list(pool.map(lambda s: download_sample(s,data_dir),manifest['samples']))
    print('All images and ground truth masks ready',flush=True)
    key=output/'key.npz'
    if not key.exists(): train_key(key)
    segmenter=TextSegmenter()
    rows=[]
    for index,(sample,data_path) in enumerate(zip(manifest['samples'],directories),1):
        directory=output/f'{index:02d}_{sample["ref_id"]}'
        print(f'[{index}/{args.count}] {sample["text"]}',flush=True)
        # Segment first so an empty prediction still receives IoU=0, never omitted.
        segmentation=segmenter.segment(data_path/'image.jpg',sample['text'],directory)
        if segmentation['selected_pixels']:
            # Reuse the saved segmentation, avoiding a second model inference.
            class SavedSegmentation:
                def segment(self,*unused,**kwargs): return dict(segmentation)
            restoration=run_image(data_path/'image.jpg',sample['text'],directory,
                                  segmenter=SavedSegmentation(),key_path=key)
            restored=True
        else:
            for stale_name in ['protected.npz','protected.preview.png','restored.png','metrics.json','comparison.png']:
                (directory/stale_name).unlink(missing_ok=True)
            restoration={}; restored=False
        target=np.array(Image.open(data_path/'ground_truth.png'))>127
        prediction=np.array(Image.open(directory/'mask.png'))>127
        score=metrics(prediction,target)
        (directory/'ground_truth.png').write_bytes((data_path/'ground_truth.png').read_bytes())
        row={k:sample[k] for k in ['image_id','ref_id','ann_id','category_id','sent_id','text']}
        row['original_text']=sample['original_text']
        row['prompt_mode']=args.prompt_mode
        row.update(score)
        try:
            saved_directory = directory.relative_to(ROOT).as_posix()
        except ValueError:
            saved_directory = str(directory)
        row.update({'directory':saved_directory,'selected_pixels':int(prediction.sum()),
                    'ground_truth_pixels':int(target.sum()),'segmentation_seconds':segmentation['segmentation_seconds'],
                    'restoration_verified':restored,'max_pixel_error':restoration.get('max_pixel_error'),
                    'mse':restoration.get('mse'),
                    'status':'ok' if restored else 'empty_prediction; IoU=0; protection skipped'})
        rows.append(row)
        (directory/'evaluation.json').write_text(json.dumps(row,indent=2))
        gallery([row],directory/'comparison.png')
        (output/'per_sample.json').write_text(json.dumps(rows,indent=2))
        print(f'  IoU={row["iou"]:.4f}, Dice={row["dice"]:.4f}, restored={restored}',flush=True)
    summary={'dataset':'RefCOCO validation subset (UNC-compatible mirror)', 'pilot_only':True,
             'prompt_mode':args.prompt_mode,
             'count':len(rows),'unique_images':len({r['image_id'] for r in rows}),'seed':args.seed,
             'threshold':0.5,'model':segmenter.model_id,'model_revision':MODEL_REVISION,
             'mean_iou':float(np.mean([r['iou'] for r in rows])),
             'mean_dice':float(np.mean([r['dice'] for r in rows])),
             'overall_iou':sum(r['intersection'] for r in rows)/sum(r['union'] for r in rows),
             'empty_predictions':sum(r['selected_pixels']==0 for r in rows),
             'exact_restorations':sum(r['restoration_verified'] for r in rows),
             'model_load_seconds':segmenter.load_seconds,
             'mean_segmentation_seconds':float(np.mean([r['segmentation_seconds'] for r in rows])),
             'notes':'Fixed images, same instance GT, no threshold tuning. Word mode uses GT category names and removes instance-disambiguation cues; not standard RefCOCO evaluation. Restoration verified on nonempty predictions only.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (output/'per_sample.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    render_galleries(rows, output)
    (output/'README.md').write_text('# RefCOCO 20-image validation pilot\n\n'+
        f'Mean IoU: {summary["mean_iou"]:.4f}\n\nMean Dice: {summary["mean_dice"]:.4f}\n\n'+
        f'Fixed seed 42; prompt mode {args.prompt_mode}; threshold 0.5; full image resolution. '+
        'Mean metrics are averaged over all samples including empty predictions. '+
        'See manifest.json for IDs, sampling rule and mirror provenance. '+
        'This is a validation pilot subset, not full benchmark evaluation.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
