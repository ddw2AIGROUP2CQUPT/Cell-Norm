"""
将图片和XML标注文件随机分割为训练集、验证集、测试集
比例：训练集:验证集:测试集 = 7:1:2
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def find_matched_pairs(image_folder: str, xml_folder: str) -> List[Tuple[Path, Path]]:
    """
    找到所有匹配的图片和XML文件对
    
    Args:
        image_folder: 图片文件夹路径
        xml_folder: XML标注文件夹路径
    
    Returns:
        匹配的文件对列表 [(image_path, xml_path), ...]
    """
    image_path = Path(image_folder)
    xml_path = Path(xml_folder)
    
    if not image_path.exists():
        raise ValueError(f"图片文件夹不存在: {image_folder}")
    if not xml_path.exists():
        raise ValueError(f"XML文件夹不存在: {xml_folder}")
    
    # 支持的图片格式
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 获取所有图片文件
    image_files = {f.stem: f for f in image_path.iterdir() 
                   if f.suffix.lower() in image_exts}
    
    # 获取所有XML文件
    xml_files = {f.stem: f for f in xml_path.iterdir() 
                 if f.suffix.lower() == '.xml'}
    
    # 找到匹配的文件对
    matched_pairs = []
    for stem in image_files:
        if stem in xml_files:
            matched_pairs.append((image_files[stem], xml_files[stem]))
        else:
            print(f"警告: 图片 {image_files[stem].name} 没有对应的XML文件，跳过")
    
    # 检查是否有XML文件没有对应的图片
    for stem in xml_files:
        if stem not in image_files:
            print(f"警告: XML文件 {xml_files[stem].name} 没有对应的图片文件，跳过")
    
    return matched_pairs


def split_data(matched_pairs: List[Tuple[Path, Path]], 
               train_ratio: float = 0.7,
               val_ratio: float = 0.1,
               test_ratio: float = 0.2,
               random_seed: int = None) -> Tuple[List[Tuple[Path, Path]], 
                                                   List[Tuple[Path, Path]], 
                                                   List[Tuple[Path, Path]]]:
    """
    将匹配的文件对随机分割为训练集、验证集、测试集
    
    Args:
        matched_pairs: 匹配的文件对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子，用于可重复性
    
    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须等于1.0，当前为: {total_ratio}")
    
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
    
    # 随机打乱
    shuffled_pairs = matched_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    total = len(shuffled_pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    # 测试集数量 = 总数 - 训练集 - 验证集（处理舍入误差）
    test_count = total - train_count - val_count
    
    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count:train_count + val_count]
    test_pairs = shuffled_pairs[train_count + val_count:]
    
    return train_pairs, val_pairs, test_pairs


def copy_files_to_split_folders(pairs: List[Tuple[Path, Path]],
                                output_image_folder: Path,
                                output_xml_folder: Path,
                                split_name: str):
    """
    将文件对复制到对应的分割文件夹
    
    Args:
        pairs: 文件对列表
        output_image_folder: 输出图片文件夹的根目录
        output_xml_folder: 输出XML文件夹的根目录
        split_name: 分割名称 ('train', 'val', 'test')
    """
    # 创建输出文件夹
    image_output = output_image_folder / split_name
    xml_output = output_xml_folder / split_name
    
    image_output.mkdir(parents=True, exist_ok=True)
    xml_output.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    for img_path, xml_path in pairs:
        # 复制图片
        shutil.copy2(img_path, image_output / img_path.name)
        # 复制XML
        shutil.copy2(xml_path, xml_output / xml_path.name)
    
    print(f"  {split_name}集: {len(pairs)} 对文件已复制到:")
    print(f"    图片: {image_output}")
    print(f"    XML: {xml_output}")


def split_dataset(image_folder: str,
                 xml_folder: str,
                 output_base_folder: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 random_seed: int = None,
                 copy_mode: bool = True):
    """
    将数据集分割为训练集、验证集、测试集
    
    Args:
        image_folder: 图片文件夹路径
        xml_folder: XML标注文件夹路径
        output_base_folder: 输出文件夹的根目录
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.2）
        random_seed: 随机种子，用于可重复性
        copy_mode: True表示复制文件，False表示移动文件
    """
    print("=" * 60)
    print("开始分割数据集")
    print("=" * 60)
    
    # 找到匹配的文件对
    print(f"\n正在查找匹配的图片和XML文件...")
    print(f"图片文件夹: {image_folder}")
    print(f"XML文件夹: {xml_folder}")
    
    matched_pairs = find_matched_pairs(image_folder, xml_folder)
    print(f"找到 {len(matched_pairs)} 对匹配的文件")
    
    if len(matched_pairs) == 0:
        print("错误: 没有找到匹配的文件对！")
        return
    
    # 分割数据
    print(f"\n正在按比例分割数据 (训练:{train_ratio}, 验证:{val_ratio}, 测试:{test_ratio})...")
    train_pairs, val_pairs, test_pairs = split_data(
        matched_pairs, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    print(f"\n分割结果:")
    print(f"  训练集: {len(train_pairs)} 对 ({len(train_pairs)/len(matched_pairs)*100:.1f}%)")
    print(f"  验证集: {len(val_pairs)} 对 ({len(val_pairs)/len(matched_pairs)*100:.1f}%)")
    print(f"  测试集: {len(test_pairs)} 对 ({len(test_pairs)/len(matched_pairs)*100:.1f}%)")
    
    # 创建输出文件夹结构
    output_base = Path(output_base_folder)
    output_images = output_base / "images"
    output_xmls = output_base / "annotations"
    
    # 复制或移动文件
    mode_str = "复制" if copy_mode else "移动"
    print(f"\n正在{mode_str}文件到输出文件夹...")
    
    if copy_mode:
        copy_func = shutil.copy2
    else:
        copy_func = shutil.move
    
    # 处理训练集
    train_image_output = output_images / "train"
    train_xml_output = output_xmls / "train"
    train_image_output.mkdir(parents=True, exist_ok=True)
    train_xml_output.mkdir(parents=True, exist_ok=True)
    
    for img_path, xml_path in train_pairs:
        copy_func(img_path, train_image_output / img_path.name)
        copy_func(xml_path, train_xml_output / xml_path.name)
    print(f"  训练集: {len(train_pairs)} 对文件已{mode_str}")
    print(f"    图片: {train_image_output}")
    print(f"    XML: {train_xml_output}")
    
    # 处理验证集
    val_image_output = output_images / "val"
    val_xml_output = output_xmls / "val"
    val_image_output.mkdir(parents=True, exist_ok=True)
    val_xml_output.mkdir(parents=True, exist_ok=True)
    
    for img_path, xml_path in val_pairs:
        copy_func(img_path, val_image_output / img_path.name)
        copy_func(xml_path, val_xml_output / xml_path.name)
    print(f"  验证集: {len(val_pairs)} 对文件已{mode_str}")
    print(f"    图片: {val_image_output}")
    print(f"    XML: {val_xml_output}")
    
    # 处理测试集
    test_image_output = output_images / "test"
    test_xml_output = output_xmls / "test"
    test_image_output.mkdir(parents=True, exist_ok=True)
    test_xml_output.mkdir(parents=True, exist_ok=True)
    
    for img_path, xml_path in test_pairs:
        copy_func(img_path, test_image_output / img_path.name)
        copy_func(xml_path, test_xml_output / xml_path.name)
    print(f"  测试集: {len(test_pairs)} 对文件已{mode_str}")
    print(f"    图片: {test_image_output}")
    print(f"    XML: {test_xml_output}")
    
    print("\n" + "=" * 60)
    print("数据分割完成！")
    print("=" * 60)
    print(f"\n输出文件夹结构:")
    print(f"  {output_base}/")
    print(f"    images/")
    print(f"      train/ ({len(train_pairs)} 张图片)")
    print(f"      val/ ({len(val_pairs)} 张图片)")
    print(f"      test/ ({len(test_pairs)} 张图片)")
    print(f"    annotations/")
    print(f"      train/ ({len(train_pairs)} 个XML文件)")
    print(f"      val/ ({len(val_pairs)} 个XML文件)")
    print(f"      test/ ({len(test_pairs)} 个XML文件)")


def main():
    # =========================
    # 配置参数（直接修改这里）
    # =========================
    image_folder = "/home/ubuntu/san/TCT/DataSets/v1-146840/spilt_x2/imgs"  # 图片文件夹路径
    xml_folder = "/home/ubuntu/san/TCT/DataSets/v1-146840/spilt_x2/label"       # XML标注文件夹路径
    output_base_folder = "/home/ubuntu/san/TCT/DataSets/v1-146840/spilt_x2/split_7_1_2"  # 输出文件夹根目录
    
    train_ratio = 0.7   # 训练集比例
    val_ratio = 0.1     # 验证集比例
    test_ratio = 0.2    # 测试集比例
    
    random_seed = 42    # 随机种子，设置为None则不固定随机种子
    copy_mode = True    # True表示复制文件（保留原文件），False表示移动文件
    
    # 执行分割
    split_dataset(
        image_folder=image_folder,
        xml_folder=xml_folder,
        output_base_folder=output_base_folder,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        copy_mode=copy_mode
    )


if __name__ == "__main__":
    main()

