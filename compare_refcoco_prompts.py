"""Compare saved original-expression and one-word RefCOCO pilot predictions."""
import csv
import json
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent

def result_path(row):
    path = Path(row['directory'])
    return path if path.is_absolute() else ROOT / path


def main():
    original_dir=ROOT/'results'/'refcoco20'
    words_dir=ROOT/'results'/'refcoco20_words'
    output=words_dir/'prompt_comparison'
    output.mkdir(parents=True,exist_ok=True)
    original=json.loads((original_dir/'per_sample.json').read_text())
    words=json.loads((words_dir/'per_sample.json').read_text())
    lookup={row['ref_id']:row for row in original}
    comparisons=[]
    for word in words:
        old=lookup[word['ref_id']]
        assert old['image_id']==word['image_id'] and old['ann_id']==word['ann_id']
        p=result_path(old);q=result_path(word)
        assert np.array_equal(np.array(Image.open(p/'ground_truth.png')),np.array(Image.open(q/'ground_truth.png')))
        assert np.array_equal(np.array(Image.open(p/'original.png')),np.array(Image.open(q/'original.png')))
        assert len(word['text'].split())==1
        comparisons.append({'ref_id':word['ref_id'],'image_id':word['image_id'],
                            'original_text':old['text'],'word':word['text'],
                            'original_iou':old['iou'],'word_iou':word['iou'],
                            'iou_change':word['iou']-old['iou'],
                            'original_dice':old['dice'],'word_dice':word['dice']})
    for start in range(0,len(words),5):
        group=words[start:start+5]
        fig,axes=plt.subplots(len(group),4,figsize=(13,3.2*len(group)),squeeze=False)
        for index,(word,axs) in enumerate(zip(group,axes)):
            old=lookup[word['ref_id']];p=result_path(old);q=result_path(word)
            for ax,file,title in zip(axs,[p/'original.png',p/'ground_truth.png',p/'mask.png',q/'mask.png'],
                                     ['Original','GT mask','Original-text prediction','One-word prediction']):
                ax.imshow(Image.open(file),cmap='gray',vmin=0,vmax=255)
                ax.set_xticks([]);ax.set_yticks([])
                if index==0:ax.set_title(title,fontsize=10)
            axs[2].set_xlabel(textwrap.fill(old['text'],36)+f'\nIoU: {old["iou"]:.3f}',fontsize=9)
            axs[3].set_xlabel(word['text']+f'\nIoU: {word["iou"]:.3f}',fontsize=9)
        fig.suptitle('Same images and instance GT | original expression vs one word',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(output/f'comparison_{start//5+1}.png',dpi=140,facecolor='white')
        plt.close(fig)
    with (output/'per_sample.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    summary={'count':len(words),'same_images_and_gt_verified':True,
             'original_mean_iou':float(np.mean([r['original_iou'] for r in comparisons])),
             'word_mean_iou':float(np.mean([r['word_iou'] for r in comparisons])),
             'original_mean_dice':float(np.mean([r['original_dice'] for r in comparisons])),
             'word_mean_dice':float(np.mean([r['word_dice'] for r in comparisons])),
             'improved_samples':sum(r['iou_change']>0 for r in comparisons),
             'worse_samples':sum(r['iou_change']<0 for r in comparisons),
             'equal_samples':sum(r['iou_change']==0 for r in comparisons),
             'note':'Word mode uses GT category names, retaining original instance masks. This changes the task and is not standard RefCOCO evaluation.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
