

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------- 常量 ----------
ABNORMAL_LABELS = {"异常", "滴虫", "菌群失调", "放线菌", "异常，菌群失调"}
CROP_SIZE = 1536
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------- XML 解析 ----------
def parse_xml(xml_path: Path) -> List[Tuple[str, int, int, int, int]]:
    """返回 [(label, xmin, ymin, xmax, ymax), ...] 仅异常框。"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall(".//object"):
            for item in obj.findall(".//item"):
                name_el = item.find("name")
                if name_el is None or name_el.text not in ABNORMAL_LABELS:
                    continue
                bnd = item.find("bndbox")
                if bnd is None:
                    continue
                boxes.append((
                    name_el.text,
                    int(bnd.find("xmin").text),
                    int(bnd.find("ymin").text),
                    int(bnd.find("xmax").text),
                    int(bnd.find("ymax").text),
                ))
        return boxes
    except Exception as e:
        print(f"解析XML失败 {xml_path}: {e}")
        return []


# ---------- XML 写出 ----------
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


# ---------- 裁剪逻辑 ----------
def compute_crop_origin(
    cx: int, cy: int, img_w: int, img_h: int, crop: int = CROP_SIZE
) -> Tuple[int, int]:
    x0 = cx - crop // 2
    y0 = cy - crop // 2
    x0 = max(0, min(x0, img_w - crop))
    y0 = max(0, min(y0, img_h - crop))
    return x0, y0


def crop_boxes_to_local(
    all_boxes: List[Tuple[str, int, int, int, int]],
    crop_x0: int,
    crop_y0: int,
    crop: int = CROP_SIZE,
    min_overlap: float = 0.3,
) -> List[Tuple[str, int, int, int, int]]:
    crop_x1, crop_y1 = crop_x0 + crop, crop_y0 + crop
    result = []
    for label, bx0, by0, bx1, by1 in all_boxes:
        ix0 = max(bx0, crop_x0)
        iy0 = max(by0, crop_y0)
        ix1 = min(bx1, crop_x1)
        iy1 = min(by1, crop_y1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        inter_area = (ix1 - ix0) * (iy1 - iy0)
        box_area = max(1, (bx1 - bx0) * (by1 - by0))
        if inter_area / box_area < min_overlap:
            continue
        lx0 = max(0, bx0 - crop_x0)
        ly0 = max(0, by0 - crop_y0)
        lx1 = min(crop, bx1 - crop_x0)
        ly1 = min(crop, by1 - crop_y0)
        result.append((label, lx0, ly0, lx1, ly1))
    return result


# ---------- 单张图处理 ----------
def process_image(
    img_path: Path,
    xml_path: Path,
    out_img_dir: Path,
    out_xml_dir: Path,
    min_overlap: float = 0.3,
) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  跳过（无法读取图片）: {img_path.name}")
        return 0
    img_h, img_w = img.shape[:2]
    if img_w < CROP_SIZE or img_h < CROP_SIZE:
        print(f"  跳过（图片 {img_w}x{img_h} 小于裁剪尺寸）: {img_path.name}")
        return 0

    all_boxes = parse_xml(xml_path)
    if not all_boxes:
        return 0

    out_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, (label, bx0, by0, bx1, by1) in enumerate(all_boxes):
        cx = (bx0 + bx1) // 2
        cy = (by0 + by1) // 2
        crop_x0, crop_y0 = compute_crop_origin(cx, cy, img_w, img_h)

        crop_img = img[crop_y0:crop_y0 + CROP_SIZE, crop_x0:crop_x0 + CROP_SIZE]
        local_boxes = crop_boxes_to_local(all_boxes, crop_x0, crop_y0, min_overlap=min_overlap)

        out_name = f"{img_path.stem}_crop{i}.png"
        out_xml_name = f"{img_path.stem}_crop{i}.xml"

        cv2.imwrite(str(out_img_dir / out_name), crop_img)
        save_xml(out_xml_dir / out_xml_name, out_name, CROP_SIZE, CROP_SIZE, local_boxes)
        count += 1

    return count


# ---------- 测试可视化 ----------
def test_visualize(
    img_path: Path,
    xml_path: Path,
    vis_dir: Path,
    min_overlap: float = 0.3,
) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  跳过（无法读取图片）: {img_path.name}")
        return 0
    img_h, img_w = img.shape[:2]
    if img_w < CROP_SIZE or img_h < CROP_SIZE:
        return 0

    all_boxes = parse_xml(xml_path)
    if not all_boxes:
        return 0

    vis_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, (label, bx0, by0, bx1, by1) in enumerate(all_boxes):
        cx = (bx0 + bx1) // 2
        cy = (by0 + by1) // 2
        crop_x0, crop_y0 = compute_crop_origin(cx, cy, img_w, img_h)

        # 左侧：原图缩略图
        scale = 640 / max(img_w, img_h)
        left = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_LINEAR)
        s = scale
        cv2.rectangle(left,
                      (int(crop_x0 * s), int(crop_y0 * s)),
                      (int((crop_x0 + CROP_SIZE) * s), int((crop_y0 + CROP_SIZE) * s)),
                      (0, 255, 0), 2)
        for j, (lbl, x0, y0, x1, y1) in enumerate(all_boxes):
            color = (0, 0, 255) if j == i else (255, 0, 0)
            cv2.rectangle(left,
                          (int(x0 * s), int(y0 * s)),
                          (int(x1 * s), int(y1 * s)),
                          color, 2)

        # 右侧：640 裁剪图 + 局部框
        crop_img = img[crop_y0:crop_y0 + CROP_SIZE, crop_x0:crop_x0 + CROP_SIZE].copy()
        local_boxes = crop_boxes_to_local(all_boxes, crop_x0, crop_y0, min_overlap=min_overlap)
        for j, (lbl, x0, y0, x1, y1) in enumerate(local_boxes):
            orig = all_boxes[i]
            is_target = (lbl == orig[0] and
                         abs(x0 - max(0, orig[1] - crop_x0)) < 5 and
                         abs(y0 - max(0, orig[2] - crop_y0)) < 5)
            color = (0, 0, 255) if is_target else (255, 0, 0)
            cv2.rectangle(crop_img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(crop_img, lbl, (x0, max(15, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        rel_cx = cx - crop_x0
        rel_cy = cy - crop_y0
        cv2.drawMarker(crop_img, (rel_cx, rel_cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        lh = left.shape[0]
        right = cv2.resize(crop_img, (lh, lh), interpolation=cv2.INTER_LINEAR)
        sep = np.full((lh, 4, 3), 180, dtype=np.uint8)
        vis = np.hstack([left, sep, right])

        cv2.putText(vis, "原图(绿=crop区,红=目标框,蓝=其余)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(vis, f"640裁剪 crop{i} | target:{label}",
                    (left.shape[1] + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

        out_name = f"{img_path.stem}_crop{i}_vis.jpg"
        cv2.imwrite(str(vis_dir / out_name), vis)
        print(f"  → {out_name}  crop_origin=({crop_x0},{crop_y0})  local_boxes={len(local_boxes)}")
        count += 1

    return count


# ---------- 主流程 ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="2048→640 按异常框裁剪")

    # 单图模式
    parser.add_argument("--image", default="/home/ubuntu/san/TCT/DataSets/v3-8037_split/images/test/1654bj005_0120.png", help="单张图片路径（与 --image-dir 二选一）")
    parser.add_argument("--xml",   default="/home/ubuntu/san/TCT/DataSets/v3-8037_split/lable/test/1654bj005_0120.xml", help="单张图片对应的 XML 路径（--image 时使用）")

    # 批量模式
    parser.add_argument("--image-dir", default="", help="原图目录")
    parser.add_argument("--xml-dir",   default="", help="XML 目录")

    parser.add_argument("--out-image", default='/home/ubuntu/san/TCT/DataSets/paper/2048_to_640_demo/1536/images', help="输出图片目录")
    parser.add_argument("--out-xml",   default='/home/ubuntu/san/TCT/DataSets/paper/2048_to_640_demo/1536/labels', help="输出 XML 目录")
    parser.add_argument("--min-overlap", type=float, default=0.3,
                        help="框在裁剪区内面积比例阈值，低于此则丢弃（默认 0.3）")
    parser.add_argument("--test", action="store_true", help="测试可视化模式")
    parser.add_argument("--test-count", type=int, default=5, help="测试模式处理图片数（批量时有效）")
    parser.add_argument("--vis-dir", default="", help="测试可视化输出目录（默认 out-image/../test_vis）")
    args = parser.parse_args()

    out_img = Path(args.out_image)
    out_xml = Path(args.out_xml)
    vis_dir = Path(args.vis_dir) if args.vis_dir else out_img.parent / "test_vis"

    # ── 单图模式 ──────────────────────────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        xml_path = Path(args.xml) if args.xml else img_path.with_suffix(".xml")
        if not img_path.exists():
            print(f"图片不存在: {img_path}")
            return
        if not xml_path.exists():
            print(f"XML 不存在: {xml_path}")
            return
        print(f"单图模式: {img_path.name}  xml={xml_path.name}")
        if args.test:
            n = test_visualize(img_path, xml_path, vis_dir, args.min_overlap)
            print(f"完成: 生成可视化图 {n} 张 → {vis_dir}")
        else:
            n = process_image(img_path, xml_path, out_img, out_xml, args.min_overlap)
            print(f"完成: 写出 {n} 张裁剪图 → {out_img}")
        return

    # ── 批量模式 ──────────────────────────────────────────────────────────────
    if not args.image_dir or not args.xml_dir:
        parser.error("请指定 --image（单图）或同时指定 --image-dir 和 --xml-dir（批量）")

    img_dir = Path(args.image_dir)
    xml_dir = Path(args.xml_dir)
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    print(f"找到图片: {len(images)} 张")

    if args.test:
        images = images[:args.test_count]
        print(f"[测试模式] 处理 {len(images)} 张，输出可视化到 {vis_dir}\n")
        total = 0
        for img_path in images:
            xml_path = xml_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                print(f"  跳过（无XML）: {img_path.name}")
                continue
            print(img_path.name)
            total += test_visualize(img_path, xml_path, vis_dir, args.min_overlap)
        print(f"\n完成: 共生成可视化图 {total} 张")
    else:
        total, skipped = 0, 0
        for img_path in images:
            xml_path = xml_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                skipped += 1
                continue
            n = process_image(img_path, xml_path, out_img, out_xml, args.min_overlap)
            total += n
            if n:
                print(f"  {img_path.name}: {n} 张裁剪")
        print(f"\n完成: 写出 {total} 张裁剪图（跳过无XML: {skipped}）")


if __name__ == "__main__":
    main()
