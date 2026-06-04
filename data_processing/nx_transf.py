

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from tqdm import tqdm

import cv2
import numpy as np

try:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    HAS_SAM = True
except Exception:
    HAS_SAM = False


# ---------- XML（自包含，原 segment_then_combine_per_image）----------
ABNORMAL_LABELS = {"异常", "滴虫", "菌群失调", "放线菌", "异常，菌群失调"}


def parse_xml(xml_path) -> List[Tuple[int, int, int, int]]:
    """解析 XML，返回异常框列表 [(xmin, ymin, xmax, ymax), ...]"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        abnormal_boxes = []
        for obj in root.findall(".//object"):
            for item in obj.findall(".//item"):
                name = item.find("name")
                if name is not None and name.text in ABNORMAL_LABELS:
                    bndbox = item.find("bndbox")
                    if bndbox is not None:
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)
                        abnormal_boxes.append((xmin, ymin, xmax, ymax))
        return abnormal_boxes
    except Exception as e:
        print(f"解析XML失败 {xml_path}: {e}")
        return []


def save_annotation_to_xml(
    xml_path: Path, image_name: str, width: int, height: int, abnormal_bboxes: List[Dict]
):
    """将多异常框写入 XML。abnormal_bboxes: [{'xmin', 'ymin', 'xmax', 'ymax'}, ...]"""
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = "3"
    obj = ET.SubElement(root, "object")
    for box in abnormal_bboxes:
        item = ET.SubElement(obj, "item")
        ET.SubElement(item, "name").text = "异常"
        bnd = ET.SubElement(item, "bndbox")
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            ET.SubElement(bnd, tag).text = str(int(box.get(tag, 0)))
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(xml_path), encoding="utf-8", xml_declaration=True)


# ---------- SAM ----------
def load_sam(model_path: str, model_type: str = "vit_h", device=None):
    if not HAS_SAM:
        raise RuntimeError("未安装 segment_anything / torch")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=model_path).to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=500,
    )
    return generator


def segment_image(sam_generator, image_rgb: np.ndarray, max_area_ratio: float = 0.3):
    """返回 (image_rgb, combined_mask) 或 (image_rgb, None)。
    max_area_ratio: mask 面积占全图比例超过此值视为背景，丢弃（默认 0.3）。
    """
    try:
        masks = sam_generator.generate(image_rgb)
        if not masks:
            return image_rgb, None
        h, w = image_rgb.shape[:2]
        total = h * w
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            if "segmentation" not in m or m["segmentation"].shape != (h, w):
                continue
            if m.get("area", total) / total > max_area_ratio:
                continue  # 面积过大，是背景 mask
            combined = np.maximum(combined, m["segmentation"].astype(np.uint8) * 255)
        return image_rgb, combined if combined.any() else None
    except Exception as e:
        print(f"分割失败: {e}")
        return image_rgb, None


def apply_segmentation(image: np.ndarray, mask: np.ndarray, bg=(0, 0, 0)) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros_like(image)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_f = (mask / 255.0).astype(np.float32)[:, :, None]
    bg_arr = np.array(bg, dtype=np.uint8)
    return (image * mask_f + (1 - mask_f) * bg_arr).astype(np.uint8)


def extract_cells(segmented: np.ndarray, border_margin: int = 10, save_border: bool = True) -> List[Dict]:
    """从分割图提取细胞列表，每项 {'image': rgb, 'bbox': (x,y,w,h), 'area': int}"""
    if segmented is None:
        return []
    bg = np.array([0, 0, 0], dtype=np.uint8)
    mask = np.any(segmented != bg, axis=2).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = segmented.shape[:2]
    cells = []
    for i in range(1, n):
        x, y, bw, bh, area = (
            int(stats[i, 0]),
            int(stats[i, 1]),
            int(stats[i, 2]),
            int(stats[i, 3]),
            int(stats[i, 4]),
        )
        if area < 100:
            continue
        on_border = (
            x < border_margin
            or y < border_margin
            or x + bw > w - border_margin
            or y + bh > h - border_margin
        )
        if on_border and not save_border:
            continue
        cell_mask = (labels == i).astype(np.uint8)
        cell_img = segmented.copy()
        cell_img[cell_mask == 0] = bg
        crop = cell_img[y : y + bh, x : x + bw]
        mc = cell_mask[y : y + bh, x : x + bw]
        crop[mc == 0] = bg
        cells.append({"image": crop, "bbox": (x, y, bw, bh), "area": area})
    return cells


def check_cell_in_abnormal(
    cell_bbox: Tuple[int, int, int, int], abnormal_boxes: List, overlap_threshold: float = 0.5
) -> bool:
    """细胞是否属于任意异常框。
    双重判据（满足其一即命中）：
      1. 交集 / 标注框面积 >= threshold（标注框被细胞覆盖，适合大细胞/小标注框）
      2. 交集 / 细胞面积   >= threshold（细胞落在标注框内，适合小细胞/大标注框）
    """
    cx, cy, cw, ch = cell_bbox
    cell_area = max(1, cw * ch)
    for (ax0, ay0, ax1, ay1) in abnormal_boxes:
        ox0 = max(cx, ax0)
        oy0 = max(cy, ay0)
        ox1 = min(cx + cw, ax1)
        oy1 = min(cy + ch, ay1)
        if ox0 < ox1 and oy0 < oy1:
            inter = (ox1 - ox0) * (oy1 - oy0)
            box_area = max(1, (ax1 - ax0) * (ay1 - ay0))
            if inter / box_area >= overlap_threshold or inter / cell_area >= overlap_threshold:
                return True
    return False


def _merge_cells_as_one(cells: List[Dict]) -> Optional[Dict]:
    """把多个已分割细胞（带原图 bbox）合成一个细胞 dict。"""
    if not cells:
        return None
    bg = np.array([0, 0, 0], dtype=np.uint8)
    x0 = min(c["bbox"][0] for c in cells)
    y0 = min(c["bbox"][1] for c in cells)
    x1 = max(c["bbox"][0] + c["bbox"][2] for c in cells)
    y1 = max(c["bbox"][1] + c["bbox"][3] for c in cells)
    w = max(1, int(x1 - x0))
    h = max(1, int(y1 - y0))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    area = 0
    for c in cells:
        cx, cy, cw, ch = c["bbox"]
        ox = int(cx - x0)
        oy = int(cy - y0)
        img = c["image"]
        m = np.any(img != bg, axis=2)
        canvas_slice = canvas[oy : oy + ch, ox : ox + cw]
        canvas_slice[m] = img[m]
        area += int(m.sum())
    return {"image": canvas, "bbox": (int(x0), int(y0), int(w), int(h)), "area": int(area)}


def merge_abnormal_cells_by_boxes(
    cells: List[Dict],
    abnormal_boxes: List[Tuple[int, int, int, int]],
    overlap_threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    if not cells or not abnormal_boxes:
        return [], list(cells)
    groups: Dict[int, List[Dict]] = {}
    abnormal_ids = set()
    for c in cells:
        hit = False
        for bi, box in enumerate(abnormal_boxes):
            if check_cell_in_abnormal(c["bbox"], [box], overlap_threshold):
                groups.setdefault(bi, []).append(c)
                hit = True
        if hit:
            abnormal_ids.add(id(c))
    abnormal_cells: List[Dict] = []
    for bi in sorted(groups.keys()):
        merged = _merge_cells_as_one(groups[bi])
        if merged is not None:
            abnormal_cells.append(merged)
    normal_cells = [c for c in cells if id(c) not in abnormal_ids]
    return abnormal_cells, normal_cells


def scale_cell(img: np.ndarray, scale: float):
    if scale == 1.0:
        return img, np.any(img != [0, 0, 0], axis=2)
    m = np.any(img != [0, 0, 0], axis=2).astype(np.uint8) * 255
    h, w = img.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    simg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    sm = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    simg[~sm] = [0, 0, 0]
    return simg, sm


def check_overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], pad: int = 1) -> bool:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x1 -= pad
    y1 -= pad
    w1 += 2 * pad
    h1 += 2 * pad
    x2 -= pad
    y2 -= pad
    w2 += 2 * pad
    h2 += 2 * pad
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


CANVAS_W = 640
CANVAS_H = 640
BG_COLOR = [253, 194, 255]


# ---------- 带 mapping 的放置（原 segment_then_combine_per_image_with_mapping 核心）----------
def _rotate_cell_with_matrix(img: np.ndarray, angle: float):
    """与 rotate_cell 一致，额外返回 2x3 仿射矩阵（src=当前图坐标 -> dst=旋转后图坐标）。"""
    if angle == 0:
        h, w = img.shape[:2]
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        mask = np.any(img != [0, 0, 0], axis=2)
        return img, mask, M, w, h
    m = np.any(img != [0, 0, 0], axis=2).astype(np.uint8) * 255
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    rimg = cv2.warpAffine(
        img, M, (nw, nh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    rmask = (
        cv2.warpAffine(m, M, (nw, nh), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    )
    rimg[~rmask] = [0, 0, 0]
    return rimg, rmask, M, nw, nh


def _prepare_patch(
    cell_img: np.ndarray,
    use_scale: bool,
    scale_range: Tuple[float, float],
    use_rotation: bool,
    max_rotation_angle: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """先 scale 再 rotate；返回 patch、mask、几何元数据。"""
    scale = 1.0
    img = cell_img
    if use_scale:
        scale = float(np.random.uniform(scale_range[0], scale_range[1]))
        img, _ = scale_cell(img, scale)
    else:
        img = cell_img
    h_after_scale, w_after_scale = img.shape[:2]
    angle = 0.0
    if use_rotation:
        angle = float(np.random.uniform(0, max_rotation_angle))
        img, cell_mask, M, cw, ch = _rotate_cell_with_matrix(img, angle)
    else:
        cell_mask = np.any(img != [0, 0, 0], axis=2)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        cw, ch = w_after_scale, h_after_scale
    meta = {
        "scale": scale,
        "angle_deg": angle,
        "after_scale_wh": [int(w_after_scale), int(h_after_scale)],
        "rotation_affine_2x3": M.tolist(),
        "patch_wh": [int(cw), int(ch)],
    }
    return img, cell_mask, meta


def place_cells_with_mapping(
    abnormal_cells: List[Dict],
    normal_cells: List[Dict],
    use_rotation: bool = True,
    use_scale: bool = True,
    max_rotation_angle: float = 360.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    margin: int = 10,
    max_attempts: int = 5000,
) -> Tuple[Optional[np.ndarray], List[Dict], List[Dict[str, Any]]]:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    placed_bboxes: List[Tuple[int, int, int, int]] = []
    placements: List[Dict[str, Any]] = []
    next_id = 0

    def try_place(cell: Dict, role: str, attempts_mul: int = 1) -> bool:
        nonlocal next_id
        sx, sy, sw, sh = (
            int(cell["bbox"][0]),
            int(cell["bbox"][1]),
            int(cell["bbox"][2]),
            int(cell["bbox"][3]),
        )
        img, cell_mask, geom = _prepare_patch(
            cell["image"], use_scale, scale_range, use_rotation, max_rotation_angle
        )
        ch, cw = img.shape[:2]
        rows = np.any(cell_mask, axis=1)
        cols = np.any(cell_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return False
        ym, yM = np.where(rows)[0][[0, -1]]
        xm, xM = np.where(cols)[0][[0, -1]]
        aw, ah = int(xM - xm + 1), int(yM - ym + 1)
        x0, y0 = margin, margin
        x1, y1 = CANVAS_W - cw - margin, CANVAS_H - ch - margin
        if x1 < x0 or y1 < y0 or cw + 2 * margin > CANVAS_W or ch + 2 * margin > CANVAS_H:
            return False
        n_try = max(1, int(max_attempts * attempts_mul))
        for _ in range(n_try):
            x = int(np.random.randint(x0, x1 + 1))
            y = int(np.random.randint(y0, y1 + 1))
            px, py = x + int(xm), y + int(ym)
            bbox = (px, py, aw, ah)
            if all(not check_overlap(bbox, p) for p in placed_bboxes):
                roi = canvas[y : y + ch, x : x + cw]
                roi[cell_mask] = img[cell_mask]
                placed_bboxes.append(bbox)
                rec = {
                    "id": next_id,
                    "role": role,
                    "source_bbox_xywh": [sx, sy, sw, sh],
                    "transform": {
                        "order": "scale_then_rotate",
                        "scale": geom["scale"],
                        "angle_deg": geom["angle_deg"],
                        "after_scale_wh": geom["after_scale_wh"],
                        "rotation_affine_2x3": geom["rotation_affine_2x3"],
                    },
                    "canvas_patch": {"x": x, "y": y, "w": cw, "h": ch},
                    "tight_in_patch_xywh": [int(xm), int(ym), aw, ah],
                    "tight_bbox_canvas_xyxy": {
                        "xmin": px,
                        "ymin": py,
                        "xmax": px + aw,
                        "ymax": py + ah,
                    },
                }
                placements.append(rec)
                next_id += 1
                return True
        return False

    abnormal_bboxes_for_xml: List[Dict] = []
    for cell in abnormal_cells:
        if try_place(cell, "abnormal", attempts_mul=2):
            b = placed_bboxes[-1]
            abnormal_bboxes_for_xml.append(
                {"xmin": b[0], "ymin": b[1], "xmax": b[0] + b[2], "ymax": b[1] + b[3]}
            )

    normal_sorted = sorted(normal_cells, key=lambda c: c["area"], reverse=True)
    remaining = list(normal_sorted)
    for _ in range(5):
        i = 0
        while i < len(remaining):
            if try_place(remaining[i], "normal", attempts_mul=1):
                remaining.pop(i)
            else:
                i += 1

    return canvas.copy(), abnormal_bboxes_for_xml, placements


# ---------- 测试拼接（原 segment_then_combine_per_image）----------
def _draw_boxes_bgr(
    img_bgr: np.ndarray,
    boxes: List,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
):
    for b in boxes:
        if isinstance(b, dict):
            x1 = int(b["xmin"])
            y1 = int(b["ymin"])
            x2 = int(b["xmax"])
            y2 = int(b["ymax"])
        else:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)


def _build_compare_image(
    orig_bgr: np.ndarray,
    combined_bgr: np.ndarray,
    orig_boxes: List,
    combined_boxes: List[Dict],
    display_height: int = 640,
    box_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    h_orig, w_orig = orig_bgr.shape[:2]
    scale = display_height / h_orig if h_orig else 1.0
    w_new = max(1, int(round(w_orig * scale)))
    orig_resized = cv2.resize(orig_bgr, (w_new, display_height), interpolation=cv2.INTER_LINEAR)
    orig_boxes_scaled = [
        (int(b[0] * scale), int(b[1] * scale), int(b[2] * scale), int(b[3] * scale)) for b in orig_boxes
    ]
    _draw_boxes_bgr(orig_resized, orig_boxes_scaled, color=box_color)
    combined_display = combined_bgr.copy()
    if combined_display.shape[0] != display_height or combined_display.shape[1] != display_height:
        combined_display = cv2.resize(combined_bgr, (display_height, display_height), interpolation=cv2.INTER_LINEAR)
    _draw_boxes_bgr(combined_display, combined_boxes, color=box_color)
    sep = np.full((display_height, 4, 3), 200, dtype=np.uint8)
    compare = np.hstack([orig_resized, sep, combined_display])
    cv2.putText(compare, "Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        compare,
        "Combined",
        (orig_resized.shape[1] + 4 + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return compare


def save_layout_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _scale_abnormal_boxes_xyxy(
    boxes: List[Tuple[int, int, int, int]], sx: float, sy: float
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        out.append(
            (
                int(round(x0 * sx)),
                int(round(y0 * sy)),
                int(round(x1 * sx)),
                int(round(y1 * sy)),
            )
        )
    return out


def _prepare_rgb_640(
    rgb,
    abnormal_boxes: List[Tuple[int, int, int, int]],
    strict_640: bool,
) -> Tuple[Any, List[Tuple[int, int, int, int]], bool, Tuple[int, int]]:
    """
    返回 (rgb_640, boxes_640, resized, raw_wh)。
    raw_wh 为读入后、缩放前的 (width, height)。
    """
    h, w = rgb.shape[:2]
    raw_wh = (int(w), int(h))
    if w == CANVAS_W and h == CANVAS_H:
        return rgb, list(abnormal_boxes), False, raw_wh
    if strict_640:
        return None, [], False, raw_wh
    sx = CANVAS_W / float(w)
    sy = CANVAS_H / float(h)
    rgb_640 = cv2.resize(rgb, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LINEAR)
    boxes_640 = _scale_abnormal_boxes_xyxy(abnormal_boxes, sx, sy)
    return rgb_640, boxes_640, True, raw_wh


def _prepare_cells_redistribute(
    img_path: Path,
    xml_file: Path,
    sam_generator,
    border_margin: int,
    overlap_threshold: float,
    save_border_cells: bool,
    strict_640: bool,
    skip_counts: Optional[Dict] = None,
    max_area_ratio: float = 0.3,
) -> Optional[
    Tuple[
        Any,
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
        List[Dict],
        List[Dict],
        bool,
        Tuple[int, int],
    ]
]:
    """
    SAM 分割 + 提细胞 + 按异常框合并（只做一次）。
    返回 (rgb_work, abnormal_work, abnormal_boxes_file, abnormal_cells, normal_cells, resized, raw_wh)。
    skip_counts: 可选的诊断计数 dict，传入后会记录每类跳过原因。
    """
    def _skip(reason: str):
        if skip_counts is not None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        return None

    try:
        if not xml_file.exists():
            return _skip("no_xml")
    except OSError as e:
        print(f"跳过 {xml_file.name}（I/O 错误: {e}）", flush=True)
        return _skip("io_error_xml")
    try:
        img = cv2.imread(str(img_path))
    except OSError as e:
        print(f"跳过 {img_path.name}（I/O 错误: {e}）", flush=True)
        return _skip("io_error_img")
    if img is None:
        return _skip("imread_none")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    abnormal_boxes = parse_xml(xml_file)
    if not abnormal_boxes:
        return _skip("no_abnormal_in_xml")
    rgb_work, abnormal_work, resized, raw_wh = _prepare_rgb_640(rgb, abnormal_boxes, strict_640)
    if rgb_work is None:
        return _skip("not_640_strict")
    if sam_generator is None:
        return _skip("no_sam")
    _, mask = segment_image(sam_generator, rgb_work, max_area_ratio=max_area_ratio)
    if mask is None:
        return _skip("sam_no_mask")
    segmented = apply_segmentation(rgb_work, mask, bg=(0, 0, 0))
    cells = extract_cells(segmented, border_margin=border_margin, save_border=save_border_cells)
    if not cells:
        return _skip("no_cells")
    abnormal_cells, normal_cells = merge_abnormal_cells_by_boxes(cells, abnormal_work, overlap_threshold)
    if not abnormal_cells:
        return _skip("no_abnormal_cells_overlap")
    return rgb_work, abnormal_work, abnormal_boxes, abnormal_cells, normal_cells, resized, raw_wh


def _build_redistributed(
    img_path: Path,
    xml_file: Path,
    sam_generator,
    border_margin: int,
    overlap_threshold: float,
    use_rotation: bool,
    use_scale: bool,
    max_rotation_angle: float,
    scale_range: Tuple[float, float],
    margin: int,
    max_attempts: int,
    save_border_cells: bool,
    strict_640: bool,
    max_area_ratio: float = 0.3,
) -> Optional[Tuple[Any, ...]]:
    """
    返回 (rgb_work, abnormal_work, abnormal_boxes_file, combined_rgb, abn_boxes, placements, resized, raw_wh)；
    任一步失败返回 None。
    """
    prep = _prepare_cells_redistribute(
        img_path,
        xml_file,
        sam_generator,
        border_margin,
        overlap_threshold,
        save_border_cells,
        strict_640,
        max_area_ratio=max_area_ratio,
    )
    if prep is None:
        return None
    rgb_work, abnormal_work, abnormal_boxes, abnormal_cells, normal_cells, resized, raw_wh = prep
    combined, abn_boxes, placements = place_cells_with_mapping(
        abnormal_cells,
        normal_cells,
        use_rotation=use_rotation,
        use_scale=use_scale,
        max_rotation_angle=max_rotation_angle,
        scale_range=scale_range,
        margin=margin,
        max_attempts=max_attempts,
    )
    if not abn_boxes:
        return None
    return rgb_work, abnormal_work, abnormal_boxes, combined, abn_boxes, placements, resized, raw_wh


def process_one_image_for_test(
    img_path: Path,
    xml_path: Path,
    sam_generator,
    border_margin: int,
    overlap_threshold: float,
    use_rotation: bool,
    use_scale: bool,
    max_rotation_angle: float,
    scale_range: Tuple[float, float],
    margin: int,
    max_attempts: int,
    save_border_cells: bool,
    strict_640: bool,
) -> Optional[Tuple[Any, Any, List[Tuple[int, int, int, int]], List[Dict]]]:
    """
    不写文件。返回 (工作原图 BGR 640, 重排后 BGR 640, 工作空间异常框 xyxy 列表, 重排后异常框 dict 列表)。
    左侧为分割所用的 640 工作图（与 XML 对齐后），右侧为随机重排结果。
    """
    xml_file = xml_path / f"{img_path.stem}.xml"
    r = _build_redistributed(
        img_path,
        xml_file,
        sam_generator,
        border_margin,
        overlap_threshold,
        use_rotation,
        use_scale,
        max_rotation_angle,
        scale_range,
        margin,
        max_attempts,
        save_border_cells,
        strict_640,
    )
    if r is None:
        return None
    rgb_work, abnormal_work, _ab_file, combined, abn_boxes, _pl, _resized, _raw = r
    orig_bgr = cv2.cvtColor(rgb_work, cv2.COLOR_RGB2BGR)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    return orig_bgr, combined_bgr, list(abnormal_work), abn_boxes


def process_one_image(
    img_path: Path,
    xml_path: Path,
    out_img_dir: Path,
    out_xml_dir: Path,
    sam_generator,
    border_margin: int,
    overlap_threshold: float,
    use_rotation: bool,
    use_scale: bool,
    max_rotation_angle: float,
    scale_range: Tuple[float, float],
    margin: int,
    max_attempts: int,
    save_border_cells: bool,
    stem_suffix: str = "",
    strict_640: bool = False,
    variants_per_image: int = 3,
    max_area_ratio: float = 0.3,
) -> int:
    """
    每张源图写出 ``variants_per_image`` 种随机重排（SAM/提细胞仅一次）。
    返回成功写出的文件组数（每组 png+xml+layout）。``variants_per_image==1`` 时文件名与旧版一致（无 _0 后缀）。
    """
    if variants_per_image < 1:
        variants_per_image = 1
    xml_file = xml_path / f"{img_path.stem}.xml"
    prep = _prepare_cells_redistribute(
        img_path,
        xml_file,
        sam_generator,
        border_margin,
        overlap_threshold,
        save_border_cells,
        strict_640,
        max_area_ratio=max_area_ratio,
    )
    if prep is None:
        return 0
    rgb_work, abnormal_work, abnormal_boxes, abnormal_cells, normal_cells, resized, raw_wh = prep

    written = 0
    out_img_dir.mkdir(parents=True, exist_ok=True)
    for v in range(variants_per_image):
        combined, abn_boxes, placements = place_cells_with_mapping(
            abnormal_cells,
            normal_cells,
            use_rotation=use_rotation,
            use_scale=use_scale,
            max_rotation_angle=max_rotation_angle,
            scale_range=scale_range,
            margin=margin,
            max_attempts=max_attempts,
        )
        if not abn_boxes:
            continue
        tag = f"{stem_suffix}_{v}" if variants_per_image > 1 else stem_suffix
        name = f"{img_path.stem}{tag}.png"
        cv2.imwrite(str(out_img_dir / name), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        xml_name = f"{img_path.stem}{tag}.xml"
        save_annotation_to_xml(out_xml_dir / xml_name, name, CANVAS_W, CANVAS_H, abn_boxes)

        layout_name = f"{img_path.stem}{tag}.layout.json"
        payload: Dict[str, Any] = {
            "version": 1,
            "pipeline": "segment_640_then_redistribute",
            "variant_index": v,
            "variants_per_image": variants_per_image,
            "source_stem": img_path.stem,
            "source_image": str(img_path.resolve()),
            "input_raw_size": {"width": raw_wh[0], "height": raw_wh[1]},
            "resized_to_640": resized,
            "strict_640_only": strict_640,
            "working_size": {"width": CANVAS_W, "height": CANVAS_H},
            "canvas": {"width": CANVAS_W, "height": CANVAS_H},
            "output_image": name,
            "output_xml": xml_name,
            "source_xml_annotation": str(xml_file.resolve()),
            "original_abnormal_boxes_xyxy_file": [
                {"xmin": int(a[0]), "ymin": int(a[1]), "xmax": int(a[2]), "ymax": int(a[3])}
                for a in abnormal_boxes
            ],
            "abnormal_boxes_xyxy_working": [
                {"xmin": int(a[0]), "ymin": int(a[1]), "xmax": int(a[2]), "ymax": int(a[3])}
                for a in abnormal_work
            ],
            "placements": placements,
            "mapping_note": "source_bbox_xywh 与逆映射均相对于 working 640 图（分割所用 rgb）；若 resized_to_640 为 true，需结合 input_raw_size 做比例换算回文件像素。",
        }
        save_layout_json(out_xml_dir / layout_name, payload)
        written += 1
    return written


_SKIP_LABELS = {
    "no_xml":                  "XML文件不存在",
    "io_error_xml":            "XML读取I/O错误",
    "io_error_img":            "图片读取I/O错误",
    "imread_none":             "图片损坏/无法解码",
    "no_abnormal_in_xml":      "XML中无<异常>标注",
    "not_640_strict":          "非640×640(strict模式)",
    "no_sam":                  "SAM未初始化",
    "sam_no_mask":             "SAM未生成任何mask",
    "no_cells":                "分割后无有效细胞(<100px²)",
    "no_abnormal_cells_overlap":"无细胞与异常框重叠",
    "exception":               "未知异常",
}

def _print_skip_summary(tag: str, total: int, ok_sources: int, skip_counts: Dict[str, int]) -> None:
    skipped = sum(skip_counts.values())
    print(f"\n[{tag}] 完成: 总图={total}, 有效源图={ok_sources}, 跳过={skipped}", flush=True)
    if skip_counts:
        for key, cnt in sorted(skip_counts.items(), key=lambda x: -x[1]):
            label = _SKIP_LABELS.get(key, key)
            print(f"  {label:30s} {cnt:>6} 张  ({cnt/total*100:.1f}%)", flush=True)


def _gpu_worker(cfg: dict) -> Tuple[int, int]:
    """
    多 GPU 子进程入口：独立加载 SAM，处理分配到的图片子集。
    cfg 中的 Path 以字符串传入（跨进程 pickle 友好）。
    返回 (ok_files, ok_sources_full)。
    """
    gpu_id = cfg["gpu_id"]
    image_paths = [Path(p) for p in cfg["image_paths"]]
    xml_dir = Path(cfg["xml_dir"])
    out_img = Path(cfg["out_img"])
    out_xml = Path(cfg["out_xml"])
    variants_per_image: int = cfg["variants_per_image"]
    num_threads: int = cfg["num_threads"]
    scale_range: Tuple[float, float] = cfg["scale_range"]

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] 加载 SAM ({cfg['sam_model_type']})...", flush=True)
    gen = load_sam(cfg["sam_model_path"], cfg["sam_model_type"], device)
    print(f"[GPU {gpu_id}] 就绪，处理 {len(image_paths)} 张", flush=True)

    ok_files = 0
    ok_sources = 0
    skip_counts: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pending: List[Tuple[int, List]] = []

        with tqdm(
            total=len(image_paths),
            desc=f"GPU:{gpu_id}",
            unit="img",
            dynamic_ncols=True,
            mininterval=0.5,
            position=gpu_id,
            leave=True,
        ) as pbar:
            for path in image_paths:
                xml_file = xml_dir / f"{path.stem}.xml"

                still_pending = []
                for idx, futs in pending:
                    if all(f.done() for f in futs):
                        n = sum(f.result() for f in futs)
                        ok_files += n
                        if n == variants_per_image:
                            ok_sources += 1
                    else:
                        still_pending.append((idx, futs))
                pending = still_pending

                try:
                    prep = _prepare_cells_redistribute(
                        path, xml_file, gen,
                        cfg["border_margin"], cfg["overlap_threshold"],
                        cfg["save_border_cells"], cfg["strict_640"],
                        skip_counts=skip_counts,
                        max_area_ratio=cfg["max_area_ratio"],
                    )
                except Exception as e:
                    print(f"\n[GPU {gpu_id}] 跳过 {path.name}: {e}", flush=True)
                    skip_counts["exception"] = skip_counts.get("exception", 0) + 1
                    pbar.update(1)
                    continue

                n_skipped = sum(skip_counts.values())
                pbar.set_postfix(written=ok_files, src_ok=ok_sources, skipped=n_skipped, refresh=False)
                pbar.update(1)
                if prep is None:
                    if n_skipped % 100 == 0 and n_skipped > 0:
                        _print_skip_summary(f"GPU:{gpu_id} 中间统计", pbar.n, ok_sources, skip_counts)
                    continue

                _, abnormal_work, abnormal_boxes, abnormal_cells, normal_cells, resized, raw_wh = prep
                out_img.mkdir(parents=True, exist_ok=True)

                futs = [
                    executor.submit(
                        _place_and_write_variant,
                        v, variants_per_image, cfg["stem_suffix"],
                        path, xml_file, out_img, out_xml,
                        abnormal_cells, normal_cells,
                        cfg["use_rotation"], cfg["use_scale"],
                        cfg["max_rotation_angle"], scale_range,
                        cfg["margin"], cfg["max_attempts"],
                        abnormal_work, abnormal_boxes, resized, raw_wh, cfg["strict_640"],
                    )
                    for v in range(variants_per_image)
                ]
                pending.append((0, futs))

        for _, futs in pending:
            n = sum(f.result() for f in futs)
            ok_files += n
            if n == variants_per_image:
                ok_sources += 1

    _print_skip_summary(f"GPU:{gpu_id}", len(image_paths), ok_sources, skip_counts)
    return ok_files, ok_sources


def _place_and_write_variant(
    v: int,
    variants_per_image: int,
    stem_suffix: str,
    img_path: Path,
    xml_file: Path,
    out_img_dir: Path,
    out_xml_dir: Path,
    abnormal_cells: List[Dict],
    normal_cells: List[Dict],
    use_rotation: bool,
    use_scale: bool,
    max_rotation_angle: float,
    scale_range: Tuple[float, float],
    margin: int,
    max_attempts: int,
    abnormal_work: List[Tuple[int, int, int, int]],
    abnormal_boxes: List[Tuple[int, int, int, int]],
    resized: bool,
    raw_wh: Tuple[int, int],
    strict_640: bool,
) -> int:
    """单个 variant 的 place+写文件，供线程池调用。返回 1 成功，0 跳过。"""
    combined, abn_boxes, placements = place_cells_with_mapping(
        abnormal_cells,
        normal_cells,
        use_rotation=use_rotation,
        use_scale=use_scale,
        max_rotation_angle=max_rotation_angle,
        scale_range=scale_range,
        margin=margin,
        max_attempts=max_attempts,
    )
    if not abn_boxes:
        return 0
    tag = f"{stem_suffix}_{v}" if variants_per_image > 1 else stem_suffix
    name = f"{img_path.stem}{tag}.png"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_dir / name), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    xml_name = f"{img_path.stem}{tag}.xml"
    save_annotation_to_xml(out_xml_dir / xml_name, name, CANVAS_W, CANVAS_H, abn_boxes)
    layout_name = f"{img_path.stem}{tag}.layout.json"
    payload: Dict[str, Any] = {
        "version": 1,
        "pipeline": "segment_640_then_redistribute",
        "variant_index": v,
        "variants_per_image": variants_per_image,
        "source_stem": img_path.stem,
        "source_image": str(img_path.resolve()),
        "input_raw_size": {"width": raw_wh[0], "height": raw_wh[1]},
        "resized_to_640": resized,
        "strict_640_only": strict_640,
        "working_size": {"width": CANVAS_W, "height": CANVAS_H},
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "output_image": name,
        "output_xml": xml_name,
        "source_xml_annotation": str(xml_file.resolve()),
        "original_abnormal_boxes_xyxy_file": [
            {"xmin": int(a[0]), "ymin": int(a[1]), "xmax": int(a[2]), "ymax": int(a[3])}
            for a in abnormal_boxes
        ],
        "abnormal_boxes_xyxy_working": [
            {"xmin": int(a[0]), "ymin": int(a[1]), "xmax": int(a[2]), "ymax": int(a[3])}
            for a in abnormal_work
        ],
        "placements": placements,
        "mapping_note": "source_bbox_xywh 与逆映射均相对于 working 640 图（分割所用 rgb）；若 resized_to_640 为 true，需结合 input_raw_size 做比例换算回文件像素。",
    }
    save_layout_json(out_xml_dir / layout_name, payload)
    return 1


def run(
    image_folder: str,
    xml_folder: str,
    output_image_folder: str,
    output_xml_folder: str,
    sam_model_path: str,
    sam_model_type: str = "vit_h",
    border_margin: int = 10,
    overlap_threshold: float = 0.5,
    use_rotation: bool = True,
    use_scale: bool = True,
    max_rotation_angle: float = 360.0,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    margin: int = 10,
    max_attempts: int = 5000,
    save_border_cells: bool = True,
    stem_suffix: str = "_redist",
    device: str = None,
    strict_640: bool = False,
    test_mode: bool = False,
    test_count: int = 10,
    compare_display_height: int = 640,
    variants_per_image: int = 3,
    num_workers: int = 4,
    gpu_ids: Optional[List[int]] = None,
    debug_dir: Optional[str] = None,
    max_area_ratio: float = 0.3,
):
    img_dir = Path(image_folder)
    xml_dir = Path(xml_folder)
    out_img = Path(output_image_folder)
    out_xml = Path(output_xml_folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    if not images:
        print("未找到图片")
        return
    scale_range = (scale_min, scale_max)
    ok_sources_full = 0
    ok_files = 0

    # 测试模式：单线程单 GPU，保持原逻辑
    if test_mode:
        images = images[:test_count]
        print(
            f"测试模式: 仅处理前 {len(images)} 张；输出「640 工作原图 | 重排结果」拼接图，不写 xml/layout"
        )
        print(f"加载 SAM: {sam_model_path}")
        gen = load_sam(sam_model_path, sam_model_type, device)
        for i, path in enumerate(images):
            result = process_one_image_for_test(
                path, xml_dir, gen, border_margin, overlap_threshold,
                use_rotation, use_scale, max_rotation_angle, scale_range,
                margin, max_attempts, save_border_cells, strict_640,
            )
            if result is not None:
                orig_bgr, combined_bgr, orig_boxes, combined_boxes = result
                compare = _build_compare_image(
                    orig_bgr, combined_bgr, orig_boxes, combined_boxes,
                    display_height=compare_display_height,
                )
                out_img.mkdir(parents=True, exist_ok=True)
                out_path = out_img / f"test_compare_640redist_{i}_{path.stem}.png"
                cv2.imwrite(str(out_path), compare)
                ok_files += 1
                print(f"  测试 {i+1}/{len(images)}: {path.name} -> {out_path.name}")
        print(f"完成: 测试对比图 {ok_files}/{len(images)}")
        return

    # 多 GPU 模式：每张卡一个子进程，各自独立加载 SAM，进程内再用线程池处理 variants
    if gpu_ids and len(gpu_ids) > 1:
        n_threads = num_workers if num_workers > 0 else (os.cpu_count() or 4)
        print(f"多 GPU 模式: gpu_ids={gpu_ids}, threads_per_gpu={n_threads}, variants={variants_per_image}")
        # 交叉分配图片，使各卡负载均衡
        chunks = [images[i::len(gpu_ids)] for i in range(len(gpu_ids))]
        base_cfg = dict(
            xml_dir=str(xml_dir), out_img=str(out_img), out_xml=str(out_xml),
            sam_model_path=sam_model_path, sam_model_type=sam_model_type,
            border_margin=border_margin, overlap_threshold=overlap_threshold,
            use_rotation=use_rotation, use_scale=use_scale,
            max_rotation_angle=max_rotation_angle, scale_range=scale_range,
            margin=margin, max_attempts=max_attempts,
            save_border_cells=save_border_cells, strict_640=strict_640,
            stem_suffix=stem_suffix, variants_per_image=variants_per_image,
            num_threads=n_threads, max_area_ratio=max_area_ratio,
        )
        worker_cfgs = [
            {**base_cfg, "gpu_id": gid, "image_paths": [str(p) for p in chunk]}
            for gid, chunk in zip(gpu_ids, chunks)
        ]
        # spawn 是 CUDA 多进程必须的启动方式
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as pool:
            futures = [pool.submit(_gpu_worker, cfg) for cfg in worker_cfgs]
            with tqdm(total=len(gpu_ids), desc="GPUs完成", unit="gpu", position=len(gpu_ids), leave=True) as pbar_gpu:
                for fut in futures:
                    n_files, n_src = fut.result()
                    ok_files += n_files
                    ok_sources_full += n_src
                    pbar_gpu.update(1)
        print(
            f"完成: 源图满 {variants_per_image} 张/张 的 {ok_sources_full}/{len(images)}，共写出 {ok_files} 个 png（含对应 xml+layout）"
        )
        return

    # 单 GPU 模式：SAM 串行（主线程），place+write 并行（线程池）
    print(f"加载 SAM: {sam_model_path}")
    gen = load_sam(sam_model_path, sam_model_type, device)
    workers = num_workers if num_workers > 0 else (os.cpu_count() or 4)
    print(f"训练模式（单 GPU）: workers={workers}, variants_per_image={variants_per_image}")

    skip_counts: Dict[str, int] = {}
    ok_prep = 0
    place_fail = 0
    dbg_dir = Path(debug_dir) if debug_dir else None
    if dbg_dir:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    # path_index -> (rgb_work BGR, segmented BGR, abnormal_work boxes)
    prep_cache: Dict[int, Any] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: List[Tuple[int, List]] = []

        def _drain_pending():
            nonlocal ok_files, ok_sources_full, place_fail
            still = []
            for idx, futs in pending:
                if all(f.done() for f in futs):
                    n = sum(f.result() for f in futs)
                    ok_files += n
                    if n > 0:
                        ok_sources_full += 1
                    else:
                        place_fail += 1
                        if dbg_dir and idx in prep_cache:
                            img_bgr, seg_bgr, abn_boxes, stem = prep_cache[idx]
                            # 工作图 + 异常框
                            vis = img_bgr.copy()
                            for bx0, by0, bx1, by1 in abn_boxes:
                                cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 0, 255), 2)
                            cv2.imwrite(str(dbg_dir / f"{stem}_work.png"), vis)
                            # 分割结果
                            cv2.imwrite(str(dbg_dir / f"{stem}_seg.png"), seg_bgr)
                    prep_cache.pop(idx, None)
                else:
                    still.append((idx, futs))
            return still

        with tqdm(total=len(images), desc="Processing", unit="img", dynamic_ncols=True) as pbar:
            for i, path in enumerate(images):
                xml_file = xml_dir / f"{path.stem}.xml"
                pending = _drain_pending()

                try:
                    prep = _prepare_cells_redistribute(
                        path, xml_file, gen, border_margin, overlap_threshold,
                        save_border_cells, strict_640,
                        skip_counts=skip_counts,
                        max_area_ratio=max_area_ratio,
                    )
                except Exception as e:
                    print(f"\n跳过 {path.name}: {e}", flush=True)
                    skip_counts["exception"] = skip_counts.get("exception", 0) + 1
                    pbar.update(1)
                    continue

                n_skipped = sum(skip_counts.values())
                pbar.set_postfix(written=ok_files, src_ok=ok_sources_full,
                                 skipped=n_skipped, place_fail=place_fail, refresh=False)
                pbar.update(1)
                if prep is None:
                    continue

                ok_prep += 1
                rgb_work, abnormal_work, abnormal_boxes, abnormal_cells, normal_cells, resized, raw_wh = prep
                out_img.mkdir(parents=True, exist_ok=True)

                if dbg_dir:
                    # 提前算出分割图用于 debug 保存（复用 rgb_work + mask）
                    _, mask = segment_image(gen, rgb_work, max_area_ratio=max_area_ratio)
                    seg_rgb = apply_segmentation(rgb_work, mask) if mask is not None else np.zeros_like(rgb_work)
                    prep_cache[i] = (
                        cv2.cvtColor(rgb_work, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR),
                        abnormal_work,
                        path.stem,
                    )

                futs = [
                    executor.submit(
                        _place_and_write_variant,
                        v, variants_per_image, stem_suffix,
                        path, xml_file, out_img, out_xml,
                        abnormal_cells, normal_cells,
                        use_rotation, use_scale, max_rotation_angle, scale_range,
                        margin, max_attempts,
                        abnormal_work, abnormal_boxes, resized, raw_wh, strict_640,
                    )
                    for v in range(variants_per_image)
                ]
                pending.append((i, futs))

        # 等待所有剩余任务完成
        for idx, futs in pending:
            n = sum(f.result() for f in futs)
            ok_files += n
            if n > 0:
                ok_sources_full += 1
                prep_cache.pop(idx, None)
            else:
                place_fail += 1
                if dbg_dir and idx in prep_cache:
                    img_bgr, seg_bgr, abn_boxes, stem = prep_cache.pop(idx)
                    vis = img_bgr.copy()
                    for bx0, by0, bx1, by1 in abn_boxes:
                        cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 0, 255), 2)
                    cv2.imwrite(str(dbg_dir / f"{stem}_work.png"), vis)
                    cv2.imwrite(str(dbg_dir / f"{stem}_seg.png"), seg_bgr)

    n_skipped_total = sum(skip_counts.values())
    print(f"\n[单GPU] 统计：总图={len(images)}", flush=True)
    print(f"  跳过（SAM/XML/无细胞等）: {n_skipped_total} 张", flush=True)
    if skip_counts:
        for key, cnt in sorted(skip_counts.items(), key=lambda x: -x[1]):
            label = _SKIP_LABELS.get(key, key)
            print(f"    {label:30s} {cnt:>6} 张  ({cnt/len(images)*100:.1f}%)", flush=True)
    print(f"  通过SAM+提细胞: {ok_prep} 张", flush=True)
    print(f"  place全部失败:  {place_fail} 张", flush=True)
    print(f"  成功写出源图:   {ok_sources_full} 张", flush=True)
    print(f"  共写出文件:     {ok_files} 个 png（含对应 xml+layout）", flush=True)
    print(f"  校验: {n_skipped_total} + {ok_prep} + 0 = {n_skipped_total + ok_prep}（应等于 {len(images)}，"
          f"其中 place_fail={place_fail} 含在 ok_prep 内）", flush=True)


def main():
    parser = argparse.ArgumentParser(description="640 图分割后随机重排 + layout.json")
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/home/ubuntu/san/TCT/DataSets/v5_1536/images/train",
        help="图片目录（建议 640×640；否则默认缩放到 640）",
    )
    parser.add_argument(
        "--xml-folder",
        type=str,
        default="/home/ubuntu/san/TCT/DataSets/v5_1536/labels/train",
        help="XML 目录",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="/home/ubuntu/san/TCT/DataSets/paper/200_pic_test_1xdata_3/images",
        help="输出图片目录",
    )
    parser.add_argument(
        "--output-xml",
        type=str,
        default="/home/ubuntu/san/TCT/DataSets/paper/200_pic_test_1xdata_3/labels",
        help="输出 XML 目录（layout.json 同目录）",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="/home/ubuntu/san/TCT/DataSets/v0-8037/model/sam_vit_h_4b8939.pth",
        help="SAM 权重路径",
    )
    parser.add_argument("--sam-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--border-margin", type=int, default=10)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--no-rotation", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--max-rotation", type=float, default=360)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--no-border-cells", action="store_true")
    parser.add_argument("--suffix", type=str, default="_redist")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--strict-640",
        action="store_true",
        help="仅处理恰好 640×640 的输入，不做缩放",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：只处理前 N 张，输出工作原图与重排图横向拼接（标异常框），不写训练用 xml/layout",
    )
    parser.add_argument("--test-count", type=int, default=10, help="测试模式下处理的图片张数")
    parser.add_argument(
        "--compare-height",
        type=int,
        default=640,
        help="拼接对比图里将左侧原图缩放到该高度（右侧同步为方形 640）",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="每张输入图随机重排生成几条样本（png+xml+layout 各一条）；1 时不加 _0 后缀",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="每张 GPU 上 place+写文件的线程数（0=自动取 CPU 核数）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
        help="多 GPU 模式：逗号分隔的 GPU ID，如 '0,1,2'；未指定则用 --device 单卡",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default='/home/ubuntu/san/TCT/DataSets/paper/200_pic_test_1xdata_3/debug',
        help="place失败时将工作图（含异常框）和分割图保存到此目录，便于排查",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.6,
        help="SAM mask 面积占全图比例超过此值视为背景并丢弃（默认 0.3）",
    )
    args = parser.parse_args()
    run(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        output_image_folder=args.output_image,
        output_xml_folder=args.output_xml,
        sam_model_path=args.sam_model,
        sam_model_type=args.sam_type,
        border_margin=args.border_margin,
        overlap_threshold=args.overlap_threshold,
        use_rotation=not args.no_rotation,
        use_scale=not args.no_scale,
        max_rotation_angle=args.max_rotation,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        margin=args.margin,
        max_attempts=args.max_attempts,
        save_border_cells=not args.no_border_cells,
        stem_suffix=args.suffix,
        device=args.device,
        strict_640=args.strict_640,
        test_mode=args.test,
        test_count=args.test_count,
        compare_display_height=args.compare_height,
        variants_per_image=max(1, args.variants),
        num_workers=args.workers,
        gpu_ids=[int(x) for x in args.gpus.split(",")] if args.gpus else None,
        debug_dir=args.debug_dir,
        max_area_ratio=args.max_area_ratio,
    )


if __name__ == "__main__":
    main()
