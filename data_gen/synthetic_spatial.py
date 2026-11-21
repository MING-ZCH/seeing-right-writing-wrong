"""Synthetic Depth Ordering and Synthetic Relative Position sets (main pool): two colored shapes on a flat background.

Rendering, item sampling and the seed (``random.seed(42)``, one shared stream) are those of the original generator.
Each loop iteration draws one relative-position scene, one depth scene and one orientation scene; the orientation
set is not part of the paper, but its draws are kept so that the two released sets come out identical.

  relative_position  "Is the <A> to the <rel> of the <B>?"; every scene gives a "yes" item (true relation) and a "no"
                     item (opposite relation), and the list is cut to ``num_per_type`` items, so 500 items use the
                     first 250 scenes (rel_pos_0000..0249.png). metadata: relation asked, object centers.
  depth_ordering     "Is the <A> in front of the <B>?"; the front shape is drawn last and always larger (radius 35-50
                     vs 20-30), with a 20-50 / 10-30 px offset, so size and occlusion are confounded (size-confounded
                     stress test). Shapes and draw order of every image go to depth_ordering_shapes.json.

-> <out-dir>/{relative_position,depth_ordering}.json, depth_ordering_shapes.json, metadata.json, images/*.png
The vLLM runners read $VTAX_DATA_ROOT/datasets/spatial_probing (absolute image paths, the default here); the copy in
data/items/spatial_probing was written with --relative-paths.

Usage: python -m data_gen.synthetic_spatial [--out-dir $VTAX_DATA_ROOT/datasets/spatial_probing] [--relative-paths]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import os
import random

from PIL import Image, ImageDraw

IMG_SIZE = (448, 448)
COLOR_MAP = {
    "red": (220, 50, 50),
    "blue": (50, 50, 220),
    "green": (50, 180, 50),
    "yellow": (220, 200, 50),
    "purple": (150, 50, 200),
    "gray": (128, 128, 128),
    "orange": (240, 140, 30),
    "cyan": (50, 200, 200),
}


def generate_shape_image(img_size: tuple[int, int], shapes: list[dict], save_path: str) -> str:
    """Draw the shapes in list order (later shapes on top) and save a PNG."""
    img = Image.new("RGB", img_size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        x, y = shape["x"], shape["y"]
        s = shape["size"]
        color = COLOR_MAP.get(shape["color"], (100, 100, 100))
        if shape["type"] == "circle":
            draw.ellipse([x - s, y - s, x + s, y + s], fill=color, outline=(0, 0, 0))
        elif shape["type"] == "square":
            draw.rectangle([x - s, y - s, x + s, y + s], fill=color, outline=(0, 0, 0))
        elif shape["type"] == "triangle":
            points = [(x, y - s), (x - s, y + s), (x + s, y + s)]
            draw.polygon(points, fill=color, outline=(0, 0, 0))
        elif shape["type"] == "diamond":
            points = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
            draw.polygon(points, fill=color, outline=(0, 0, 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, quality=85)
    return save_path


def generate(num_per_type: int, out_dir: str, relative_paths: bool = False) -> dict[str, list]:
    """Items of both sets (images are written to out_dir/images)."""
    results = {"relative_position": [], "depth_ordering": [], "depth_ordering_shapes": []}
    random.seed(42)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    ref = (lambda fn: f"images/{fn}") if relative_paths else (lambda fn: os.path.join(img_dir, fn))

    shape_types = ["circle", "square", "triangle", "diamond"]
    colors = ["red", "blue", "green", "yellow", "purple", "gray", "orange", "cyan"]
    img_size = IMG_SIZE
    margin = 60
    center = img_size[0] // 2

    for i in range(num_per_type):
        type1, type2 = random.sample(shape_types, 2)
        color1, color2 = random.sample(colors, 2)
        size1 = random.randint(25, 45)
        size2 = random.randint(25, 45)

        # relative position
        relation = random.choice(["left", "right", "above", "below"])
        if relation == "left":
            x1 = random.randint(margin, center - 30)
            x2 = random.randint(center + 30, img_size[0] - margin)
            y1 = random.randint(margin + 40, img_size[1] - margin - 40)
            y2 = random.randint(margin + 40, img_size[1] - margin - 40)
        elif relation == "right":
            x1 = random.randint(center + 30, img_size[0] - margin)
            x2 = random.randint(margin, center - 30)
            y1 = random.randint(margin + 40, img_size[1] - margin - 40)
            y2 = random.randint(margin + 40, img_size[1] - margin - 40)
        elif relation == "above":
            x1 = random.randint(margin + 40, img_size[0] - margin - 40)
            x2 = random.randint(margin + 40, img_size[0] - margin - 40)
            y1 = random.randint(margin, center - 30)
            y2 = random.randint(center + 30, img_size[1] - margin)
        else:  # below
            x1 = random.randint(margin + 40, img_size[0] - margin - 40)
            x2 = random.randint(margin + 40, img_size[0] - margin - 40)
            y1 = random.randint(center + 30, img_size[1] - margin)
            y2 = random.randint(margin, center - 30)
        shapes = [
            {"type": type1, "color": color1, "x": x1, "y": y1, "size": size1},
            {"type": type2, "color": color2, "x": x2, "y": y2, "size": size2},
        ]
        fn = f"rel_pos_{i:04d}.png"
        if 2 * i < num_per_type:  # scenes beyond the cut are not referenced by any item
            generate_shape_image(img_size, shapes, os.path.join(img_dir, fn))
        results["relative_position"].append({
            "image_path": ref(fn),
            "question": f"Is the {color1} {type1} to the {relation} of the {color2} {type2}?",
            "answer": "yes",
            "task_type": "relative_position",
            "spatial_label": 1,
            "source": "synthetic",
            "metadata": {"relation": relation, "obj1_pos": [x1, y1], "obj2_pos": [x2, y2]},
        })
        wrong_relation = {"left": "right", "right": "left", "above": "below", "below": "above"}[relation]
        results["relative_position"].append({
            "image_path": ref(fn),
            "question": f"Is the {color1} {type1} to the {wrong_relation} of the {color2} {type2}?",
            "answer": "no",
            "task_type": "relative_position",
            "spatial_label": 0,
            "source": "synthetic",
            "metadata": {"relation": wrong_relation, "obj1_pos": [x1, y1], "obj2_pos": [x2, y2]},
        })

        # depth ordering: the front shape is drawn last and larger
        front_obj = random.choice([1, 2])
        cx = random.randint(center - 60, center + 60)
        cy = random.randint(center - 60, center + 60)
        if front_obj == 1:
            s_front = random.randint(35, 50)
            s_back = random.randint(20, 30)
            offset_x = random.randint(20, 50) * random.choice([-1, 1])
            offset_y = random.randint(10, 30) * random.choice([-1, 1])
            shapes_depth = [
                {"type": type2, "color": color2, "x": cx + offset_x, "y": cy + offset_y, "size": s_back},
                {"type": type1, "color": color1, "x": cx, "y": cy, "size": s_front},
            ]
            label_depth = 1
        else:
            s_front = random.randint(35, 50)
            s_back = random.randint(20, 30)
            offset_x = random.randint(20, 50) * random.choice([-1, 1])
            offset_y = random.randint(10, 30) * random.choice([-1, 1])
            shapes_depth = [
                {"type": type1, "color": color1, "x": cx + offset_x, "y": cy + offset_y, "size": s_back},
                {"type": type2, "color": color2, "x": cx, "y": cy, "size": s_front},
            ]
            label_depth = 0
        fn = f"depth_{i:04d}.png"
        generate_shape_image(img_size, shapes_depth, os.path.join(img_dir, fn))
        results["depth_ordering"].append({
            "image_path": ref(fn),
            "question": f"Is the {color1} {type1} in front of the {color2} {type2}?",
            "answer": "yes" if label_depth == 1 else "no",
            "task_type": "depth_ordering",
            "spatial_label": label_depth,
            "source": "synthetic",
        })
        results["depth_ordering_shapes"].append({"image": fn, "shapes_back_to_front": shapes_depth})

        # orientation scene of the original generator: draws only (keeps the random stream identical)
        quadrant = random.choice([0, 1, 2, 3])
        if quadrant == 0:
            random.randint(margin, center - 30), random.randint(margin, center - 30)
        elif quadrant == 1:
            random.randint(center + 30, img_size[0] - margin), random.randint(margin, center - 30)
        elif quadrant == 2:
            random.randint(margin, center - 30), random.randint(center + 30, img_size[1] - margin)
        else:
            random.randint(center + 30, img_size[0] - margin), random.randint(center + 30, img_size[1] - margin)

    results["relative_position"] = results["relative_position"][:num_per_type]
    results["depth_ordering"] = results["depth_ordering"][:num_per_type]
    return results


def save(results: dict[str, list], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, samples in results.items():
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(samples, f, indent=2, default=str)
    meta = {"samples_per_task": {k: len(results[k]) for k in ("relative_position", "depth_ordering")},
            "image_size": list(IMG_SIZE), "generation_seed": 42,
            "note": "orientation scenes of the original generator are drawn (random stream) but not written"}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the Synthetic Depth Ordering / Relative Position sets (seed 42).")
    ap.add_argument("--out-dir", default=f"{_vtp.DATASETS}/spatial_probing")
    ap.add_argument("--num-per-type", type=int, default=500)
    ap.add_argument("--relative-paths", action="store_true", help="store image paths relative to --out-dir")
    a = ap.parse_args()
    res = generate(a.num_per_type, a.out_dir, a.relative_paths)
    save(res, a.out_dir)
    print(f"wrote {len(res['relative_position'])} relative-position and {len(res['depth_ordering'])} depth items "
          f"to {a.out_dir}")


if __name__ == "__main__":
    main()
