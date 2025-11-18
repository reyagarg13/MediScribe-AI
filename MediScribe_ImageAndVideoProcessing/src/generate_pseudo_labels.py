#!/usr/bin/env python3
"""Generate pseudo labels using EasyOCR. Outputs a JSONL with fields:
image_path, text, conf
"""
import os, json, argparse
from tqdm import tqdm

def main(images_dir, out_file, use_gpu=True):
    try:
        import easyocr
    except Exception as e:
        print('Please install easyocr: pip install easyocr')
        raise
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    results = []
    for fn in tqdm(sorted(os.listdir(images_dir))):
        if not fn.lower().endswith(('.png','.jpg','.jpeg','.webp')):
            continue
        path = os.path.join(images_dir, fn)
        try:
            res = reader.readtext(path, detail=1)
        except Exception as e:
            print('EasyOCR failed for', path, '->', e)
            res = []
        text = "\n".join([r[1] for r in res]) if res else ""
        conf = float(sum([r[2] for r in res]) / max(1, len(res))) if res else 0.0
        results.append({"image_path": path, "text": text, "conf": conf})
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('Saved', len(results), 'pseudo-labels to', out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out', dest='out_file', required=True)
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()
    main(args.images_dir, args.out_file, use_gpu=not args.no_gpu)
