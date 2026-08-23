from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CHANGES = HERE / "results" / "manifest_changes.csv"
TRAIN = WORKSPACE / "BDC2026" / "train"
OUTPUT = HERE / "swap_contact_sheets"

CLASS_NAMES = {0: "Recyclable", 1: "Electronic", 2: "Organic"}
COLS, ROWS = 6, 5
CELL_WIDTH, IMAGE_HEIGHT, LABEL_HEIGHT = 220, 190, 42
HEADER_HEIGHT = 46


def natural_key(filename: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)_(\d+)\.[^.]+", filename)
    return (match.group(1).upper(), int(match.group(2))) if match else (filename, -1)


def image_index() -> dict[str, Path]:
    return {path.name: path for path in TRAIN.rglob("*") if path.is_file()}


def load_changes() -> list[dict[str, object]]:
    with CHANGES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["old_label"] = int(row["old_label"])
        row["new_label"] = int(row["new_label"])
    assert len(rows) == 514
    return sorted(
        rows,
        key=lambda row: (row["old_label"], row["new_label"], natural_key(row["filename"])),
    )


def draw_sheet(
    rows: list[dict[str, object]],
    paths: dict[str, Path],
    old_label: int,
    new_label: int,
    page: int,
    pages: int,
) -> tuple[Path, list[dict[str, object]]]:
    cell_height = IMAGE_HEIGHT + LABEL_HEIGHT
    canvas = Image.new(
        "RGB",
        (COLS * CELL_WIDTH, HEADER_HEIGHT + ROWS * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    title_font = ImageFont.load_default(size=20)
    title = (
        f"{old_label} {CLASS_NAMES[old_label]} -> "
        f"{new_label} {CLASS_NAMES[new_label]} | page {page}/{pages} | {len(rows)} images"
    )
    draw.text((12, 11), title, fill="black", font=title_font)

    index_rows = []
    for position, row in enumerate(rows):
        grid_row, column = divmod(position, COLS)
        x = column * CELL_WIDTH
        y = HEADER_HEIGHT + grid_row * cell_height
        path = paths[row["filename"]]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = ImageOps.contain(
                image,
                (CELL_WIDTH - 10, IMAGE_HEIGHT - 10),
                Image.Resampling.LANCZOS,
            )
        left = x + (CELL_WIDTH - image.width) // 2
        top = y + (IMAGE_HEIGHT - image.height) // 2
        canvas.paste(image, (left, top))
        draw.rectangle(
            (x, y, x + CELL_WIDTH - 1, y + cell_height - 1),
            outline=(150, 150, 150),
        )
        caption = f"{row['filename']}\n{old_label} -> {new_label}"
        draw.multiline_text((x + 6, y + IMAGE_HEIGHT + 3), caption, fill="black", font=font, spacing=1)
        index_rows.append(
            {
                "cell": position + 1,
                "row": grid_row + 1,
                "column": column + 1,
                "filename": row["filename"],
                "old_label": old_label,
                "old_class": CLASS_NAMES[old_label],
                "new_label": new_label,
                "new_class": CLASS_NAMES[new_label],
                "image_path": str(path),
            }
        )

    name = (
        f"{old_label}_{CLASS_NAMES[old_label].lower()}_to_"
        f"{new_label}_{CLASS_NAMES[new_label].lower()}_p{page:02d}.jpg"
    )
    output = OUTPUT / name
    canvas.save(output, quality=94, subsampling=0)
    for row in index_rows:
        row["sheet"] = name
    return output, index_rows


def main() -> None:
    changes = load_changes()
    paths = image_index()
    missing = sorted({row["filename"] for row in changes} - paths.keys())
    if missing:
        raise FileNotFoundError(f"Missing swapped images: {missing}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_sheet = COLS * ROWS
    all_index_rows = []
    generated = []
    directions = sorted({(row["old_label"], row["new_label"]) for row in changes})
    for old_label, new_label in directions:
        group = [
            row
            for row in changes
            if row["old_label"] == old_label and row["new_label"] == new_label
        ]
        pages = (len(group) + per_sheet - 1) // per_sheet
        for page, start in enumerate(range(0, len(group), per_sheet), 1):
            output, index_rows = draw_sheet(
                group[start : start + per_sheet],
                paths,
                old_label,
                new_label,
                page,
                pages,
            )
            generated.append(output)
            all_index_rows.extend(index_rows)

    index_path = OUTPUT / "swap_contact_sheet_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sheet", "cell", "row", "column", "filename",
                "old_label", "old_class", "new_label", "new_class", "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(all_index_rows)

    assert len(all_index_rows) == 514
    print(f"514 swaps -> {len(generated)} sheets")
    print(index_path)


if __name__ == "__main__":
    main()
