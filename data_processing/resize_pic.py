"""
将 1024×1024 裁剪图 resize 到 640×640，同步缩放 XML 中的异常框坐标。

用法：
  python resize_1024_to_640.py \
    --image-dir /data/cropped_1024/images \
    --xml-dir   /data/cropped_1024/labels \
    --out-image /data/resized_640/images \
    --out-xml   /data/resized_640/labels
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import cv2

SRC_SIZE = 1536
DST_SIZE = 640
SCALE = DST_SIZE / SRC_SIZE  # 0.625
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_xml(xml_path: Path) -> Tuple[List[Tuple[str, int, int, int, int]], ET.Element]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall(".//object"):
        for item in obj.findall(".//item"):
            name_el = item.find("name")
            bnd = item.find("bndbox")
            if name_el is None or bnd is None:
                continue
            boxes.append((
                name_el.text,
                int(bnd.find("xmin").text),
                int(bnd.find("ymin").text),
                int(bnd.find("xmax").text),
                int(bnd.find("ymax").text),
            ))
    return boxes, root


def save_xml(
    xml_path: Path,
    image_name: str,
    width: int,
    height: int,
    boxes: List[Tuple[str, int, int, int, int]],
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = "3"
    obj = ET.SubElement(root, "object")
    for label, x0, y0, x1, y1 in boxes:
        item = ET.SubElement(obj, "item")
        ET.SubElement(item, "name").text = label
        bnd = ET.SubElement(item, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x0)
        ET.SubElement(bnd, "ymin").text = str(y0)
        ET.SubElement(bnd, "xmax").text = str(x1)
        ET.SubElement(bnd, "ymax").text = str(y1)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(xml_path), encoding="utf-8", xml_declaration=True)


def scale_boxes(
    boxes: List[Tuple[str, int, int, int, int]], scale: float, dst: int
) -> List[Tuple[str, int, int, int, int]]:
    result = []
    for label, x0, y0, x1, y1 in boxes:
        nx0 = max(0, min(int(round(x0 * scale)), dst - 1))
        ny0 = max(0, min(int(round(y0 * scale)), dst - 1))
        nx1 = max(0, min(int(round(x1 * scale)), dst))
        ny1 = max(0, min(int(round(y1 * scale)), dst))
        if nx1 > nx0 and ny1 > ny0:
            result.append((label, nx0, ny0, nx1, ny1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="1024→640 resize + XML 坐标同步缩放")
    parser.add_argument(
        "--image-dir",
        default="/home/ubuntu/san/TCT/DataSets/v5_1536/images/test",
    )
    parser.add_argument(
        "--xml-dir",
        default="/home/ubuntu/san/TCT/DataSets/v5_1536/labels/test",
    )
    parser.add_argument(
        "--out-image",
        default="/home/ubuntu/san/TCT/DataSets/v5_1536/1536_to_640/test/images",
    )
    parser.add_argument(
        "--out-xml",
        default="/home/ubuntu/san/TCT/DataSets/v5_1536/1536_to_640/test/labels",
    )
    parser.add_argument(
        "--src-size", type=int, default=SRC_SIZE, help="源图尺寸（默认 1024）"
    )
    parser.add_argument(
        "--dst-size", type=int, default=DST_SIZE, help="目标尺寸（默认 640）" 
    )
    args = parser.parse_args()

    scale = args.dst_size / args.src_size
    img_dir = Path(args.image_dir)
    xml_dir = Path(args.xml_dir)
    out_img = Path(args.out_image)
    out_xml = Path(args.out_xml)

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    print(f"找到图片: {len(images)} 张  scale={scale:.4f} ({args.src_size}→{args.dst_size})")

    ok = skipped = 0
    for img_path in images:
        xml_path = xml_dir / f"{img_path.stem}.xml"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  跳过（无法读取）: {img_path.name}")
            skipped += 1
            continue

        resized = cv2.resize(img, (args.dst_size, args.dst_size), interpolation=cv2.INTER_LINEAR)

        out_img.mkdir(parents=True, exist_ok=True)
        out_name = img_path.stem + ".png"
        cv2.imwrite(str(out_img / out_name), resized)

        if xml_path.exists():
            boxes, _ = parse_xml(xml_path)
            scaled_boxes = scale_boxes(boxes, scale, args.dst_size)
            save_xml(
                out_xml / f"{img_path.stem}.xml",
                out_name,
                args.dst_size,
                args.dst_size,
                scaled_boxes,
            )
        else:
            print(f"  警告（无XML）: {img_path.name}")

        ok += 1

    print(f"\n完成: 处理 {ok} 张，跳过 {skipped} 张")


if __name__ == "__main__":
    main()
