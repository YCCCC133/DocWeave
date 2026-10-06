"""Conservative exam-question grouping before joint text/LaTeX recognition.

No model is loaded.  OCR prefixes identify printed question/option anchors only;
their recognized body is never copied into a merged block or used to infer math.
The full question is recognized by the existing Text Recognition task, so a B/D
continuation that wraps to the left stays in its original geometric context.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any
import time

_QUESTION_START = re.compile(r"^\s*[（(]\s*(\d{1,3})\s*[）)]")
_QUESTION_ANY = re.compile(r"(?<![A-Za-z0-9])[（(]\s*(\d{1,3})\s*[）)]")
_OPTION = re.compile(r"(?<![A-Za-z0-9])[（(]\s*([ABCD])\s*[）)]")
_TEXT_TYPES = frozenset({"text", "vertical_text"})
_MERGE_TYPES = _TEXT_TYPES | {"equation", "display_formula", "inline_formula", "formula_number"}


@dataclass(frozen=True)
class MergeResult:
    blocks: list[dict[str, Any]]
    groups: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PageMergeResult:
    blocks_list: list[list[dict[str, Any]]]
    anchor_pages: list[list[dict[str, Any]]]
    groups: list[tuple[dict[str, Any], ...]]
    rejected: list[tuple[dict[str, Any], ...]]
    errors: tuple[dict[str, Any], ...]
    timings: dict[str, Any]


def _bbox(value: Sequence[Any]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("bbox must contain four coordinates")
    box = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox must be finite and nonempty")
    return box


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _overlap(a, b) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 1e-7 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-7


def _anchors(prefix: Any) -> tuple[int | None, list[str], bool]:
    """Accept strings or explicitly reduced marker evidence; ignore body values."""
    if isinstance(prefix, Mapping):
        question = prefix.get("question")
        if question is not None and (not isinstance(question, int) or isinstance(question, bool) or not 1 <= question <= 999):
            raise ValueError("question anchor must be an integer from 1 to 999")
        options = list(prefix.get("options", ()))
        if any(value not in "ABCD" or len(value) != 1 for value in options):
            raise ValueError("option anchors must be A/B/C/D")
        return question, options, bool(prefix.get("ambiguous", False))
    if prefix is None:
        return None, [], False
    if not isinstance(prefix, str):
        raise TypeError("each prefix must be a string, mapping, or None")
    question_match = _QUESTION_START.match(prefix)
    question = int(question_match.group(1)) if question_match else None
    if question == 0:
        question = None
    options = _OPTION.findall(prefix)
    # More than one numbered question in one OCR region cannot be separated by
    # geometry here.  Do not invent internal question coordinates.
    seen_questions = _QUESTION_ANY.findall(prefix)
    return question, options, len(set(seen_questions)) > 1


def attach_anchor_evidence(blocks, prefixes):
    """Copy source blocks and attach only explicit anchor/geometry evidence."""
    if len(blocks) != len(prefixes):
        raise ValueError("blocks and anchor prefixes count mismatch")
    output = deepcopy([dict(block) for block in blocks])
    for source_index, (block, prefix) in enumerate(zip(output, prefixes)):
        question, options, ambiguous = _anchors(prefix)
        source = prefix if isinstance(prefix, Mapping) else {}
        evidence = {"question": question, "options": options, "ambiguous": ambiguous,
                    "anchors": deepcopy(list(source.get("anchors", ()))), "source_indices": [source_index]}
        evidence["line_boxes_px"] = deepcopy(list(source.get("line_boxes_px", ())))
        evidence["line_boxes_unit"] = deepcopy(list(source.get("line_boxes_unit", ())))
        if "roi_bbox_px" in source:
            evidence["roi_bbox_px"] = deepcopy(source["roi_bbox_px"])
        block["_converter_anchor_evidence"] = evidence
        block["_converter_source_index"] = source_index
        block["_converter_source_indices"] = [source_index]
    return output


def pad_text_blocks_safely(blocks, page_size, *, padding_px=12.0, units="unit"):
    """Pad text crops up to 12px, sharing gaps with neighbours without new overlap.

    Original overlaps are left in place and do not grow.  Disjoint original
    boxes that would overlap after padding divide the empty gap at its midpoint.
    A padding adjustment never shrinks or omits any of the original source bbox.
    ``_converter_unpadded_bbox`` lets coverage retries refer to the exact input.
    """
    if padding_px < 0 or page_size[0] <= 0 or page_size[1] <= 0 or units not in {"unit", "pixel"}:
        raise ValueError("invalid padding, image size, or bbox units")
    output = deepcopy([dict(block) for block in blocks])
    width, height = page_size
    def pixels(box):
        return (box[0] * width, box[1] * height, box[2] * width, box[3] * height) if units == "unit" else box
    bases = [pixels(_bbox(block["bbox"])) for block in output]
    padded = [(max(0.0, b[0] - padding_px), max(0.0, b[1] - padding_px),
               min(float(width), b[2] + padding_px), min(float(height), b[3] + padding_px))
              if block.get("type", block.get("label")) in _TEXT_TYPES else b
              for block, b in zip(output, bases)]
    final_boxes = []
    for index, (block, base, candidate) in enumerate(zip(output, bases, padded)):
        if block.get("type", block.get("label")) not in _TEXT_TYPES:
            final_boxes.append(base)
            continue
        left, top, right, bottom = candidate
        for neighbour_index, (neighbour, neighbour_pad) in enumerate(zip(bases, padded)):
            if neighbour_index == index or not _overlap(candidate, neighbour_pad):
                continue
            if _overlap(base, neighbour):
                # Do not enlarge an existing overlap; the original crop stays.
                left, top, right, bottom = base
                break
            if neighbour[2] <= base[0]:
                left = max(left, (neighbour[2] + base[0]) / 2)
            if neighbour[0] >= base[2]:
                right = min(right, (base[2] + neighbour[0]) / 2)
            if neighbour[3] <= base[1]:
                top = max(top, (neighbour[3] + base[1]) / 2)
            if neighbour[1] >= base[3]:
                bottom = min(bottom, (base[3] + neighbour[1]) / 2)
        padded_box = (left, top, right, bottom)
        assert left <= base[0] and top <= base[1] and right >= base[2] and bottom >= base[3]
        block["_converter_unpadded_bbox"] = deepcopy(block["bbox"])
        block["_converter_padding_px"] = [base[0] - left, base[1] - top, right - base[2], bottom - base[3]]
        block["bbox"] = [left / width, top / height, right / width, bottom / height] if units == "unit" else list(padded_box)
        final_boxes.append(padded_box)
    # Source boxes can already overlap, but this helper never creates a new pair.
    for index, box in enumerate(final_boxes):
        for previous, previous_box in enumerate(final_boxes[:index]):
            if not _overlap(bases[index], bases[previous]):
                assert not _overlap(box, previous_box), "padding created a neighbouring crop overlap"
    return output


def merge_question_blocks(
    blocks: Sequence[Mapping[str, Any]],
    prefixes: Sequence[Any],
    page_size: tuple[int, int],
    *,
    units: str = "unit",
    margin_px: float = 4.0,
) -> MergeResult:
    """Return copied blocks; validated question runs replace their source blocks.

    The caller supplies PP layout blocks in reading order plus OCR marker
    evidence in exactly the same order.  ``units='unit'`` matches MinerU high
    VLM blocks; ``units='pixel'`` matches raw PP boxes.  Margins are always pixels.
    Only a bounded question followed by its next consecutive question is merged.
    The last question or incomplete page continuation remains as original blocks.

    Titles, tables, images, captions, headers, and ordinary unnumbered text keep
    their original objects' data and order.  Any excluded block intersecting a
    proposed crop cancels that group; it cannot be recognized twice.  Rotations,
    column jumps, missing labels, reversed question order, or ambiguous prefixes
    also leave the original blocks untouched.
    """
    if len(blocks) != len(prefixes):
        raise ValueError("blocks and OCR prefixes must have identical length")
    width, height = page_size
    if width <= 0 or height <= 0 or margin_px < 0:
        raise ValueError("invalid page size or margin")
    if units not in {"unit", "pixel"}:
        raise ValueError("units must be 'unit' or 'pixel'")
    original = deepcopy([dict(block) for block in blocks])
    data = [_anchors(prefix) for prefix in prefixes]
    raw_boxes = [_bbox(block["bbox"]) for block in original]

    def pixels(box):
        return (box[0] * width, box[1] * height, box[2] * width, box[3] * height) if units == "unit" else box

    # Line extents can exceed a rounded block bbox by a pixel; preserve them.
    boxes = [pixels(_union([box, *(_bbox(line["bbox"]) for line in block.get("lines", ()))]))
             for block, box in zip(original, raw_boxes)]
    for index, prefix in enumerate(prefixes):
        source = prefix if isinstance(prefix, Mapping) else {}
        # Detection lines may reach beyond a rounded PP bbox.  Keep their exact
        # source extents in a validated joint crop; unmerged input stays intact.
        line_boxes = [_bbox(box) for box in source.get("line_boxes_px", ())]
        if line_boxes:
            boxes[index] = _union([boxes[index], *line_boxes])
    anchors = [(index, question) for index, (question, _, ambiguous) in enumerate(data)
               if question is not None and original[index].get("type", original[index].get("label")) in _TEXT_TYPES]
    groups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    proposals: dict[int, tuple[tuple[int, ...], dict[str, Any]]] = {}
    consumed: set[int] = set()
    for (start, question), (end, next_question) in zip(anchors, anchors[1:]):
        def reject(reason):
            rejected.append({"question": question, "source_start": start, "source_end_exclusive": end, "reason": reason})

        if next_question != question + 1:
            reject("question numbers are not consecutive")
            continue
        members = tuple(index for index in range(start, end)
                        if original[index].get("type", original[index].get("label")) in _MERGE_TYPES)
        if len(members) < 2:
            reject("not a multi-block question")
            continue
        if any(data[index][2] for index in range(start, end)):
            reject("one OCR region contains ambiguous numbered anchors")
            continue
        option_sequence = [label for index in range(start, end) for label in data[index][1]]
        if option_sequence != list("ABCD"):
            reject("requires one A/B/C/D sequence in original order")
            continue
        if any(original[index].get("angle", 0) not in {None, 0} for index in members):
            reject("rotated question blocks")
            continue
        first, following = boxes[start], boxes[end]
        # A column transition starts above or to the side of the previous one.
        # It is never converted into a page-wide reading-order guess.
        if following[1] <= first[1] or abs(following[0] - first[0]) > max(width * 0.08, 20.0):
            reject("question boundary changes column or reading direction")
            continue
        merged = _union([boxes[index] for index in members])
        if merged[3] >= following[1] or merged[1] < first[1] - max(margin_px, 2.0):
            reject("source boxes cross the next question or extend above the stem")
            continue
        expanded = (max(0.0, merged[0] - margin_px), max(0.0, merged[1] - margin_px),
                    min(float(width), merged[2] + margin_px), min(float(height), merged[3] + margin_px))
        if any(_overlap(expanded, box) for index, box in enumerate(boxes) if index not in members):
            reject("joint crop would include an independent block or another question")
            continue
        if any(index in consumed for index in members):
            reject("source block already belongs to a question")
            continue
        new_block = deepcopy(original[start])
        new_block.update({"type": "text", "bbox": [expanded[0] / width, expanded[1] / height,
                                                    expanded[2] / width, expanded[3] / height]
                          if units == "unit" else list(expanded), "angle": 0, "merge_prev": False})
        # OCR content only supplied anchor evidence.  The VLM will recognize the
        # source crop; no OCR body, partial output, or old line metadata survives.
        for key in ("content", "lines", "score", "cell_merge"):
            new_block.pop(key, None)
        new_block["_converter_joint_question"] = question
        new_block["_converter_joint_sources"] = list(members)
        new_block["_converter_source_index"] = start
        new_block["_converter_source_indices"] = list(members)
        evidence = [original[index].get("_converter_anchor_evidence", {}) for index in members]
        new_block["_converter_anchor_evidence"] = {
            "question": question, "options": option_sequence, "ambiguous": False,
            "anchors": [deepcopy(anchor) for item in evidence for anchor in item.get("anchors", ())],
            "line_boxes_px": [deepcopy(box) for item in evidence for box in item.get("line_boxes_px", ())],
            "line_boxes_unit": [deepcopy(box) for item in evidence for box in item.get("line_boxes_unit", ())],
            "source_indices": list(members),
        }
        proposals[start] = (members, new_block)
        consumed.update(members)
        groups.append({"question": question, "source_indices": list(members), "bbox_px": list(expanded),
                       "option_anchors": option_sequence, "margin_px": margin_px})
    result = []
    for index, block in enumerate(original):
        if index in proposals:
            result.append(proposals[index][1])
        elif index not in consumed:
            result.append(block)
    return MergeResult(result, tuple(groups), tuple(rejected))


def merge_page_question_regions(
    images_pil_list: Sequence[Any],
    high_vlm_blocks: Sequence[Sequence[Mapping[str, Any]]],
    ocr_model: Any,
    *,
    run_ocr_inference: Any = None,
    line_cropper: Any = None,
    margin_px: float = 4.0,
    probe_margin_px: float = 8.0,
    min_anchor_score: float = 0.60,
    max_lines_per_roi: int | None = None,
) -> PageMergeResult:
    """Detect per-ROI lines; batch-recognize all crops with the existing PP OCR.

    PytorchPaddleOCR.ocr does NOT accept a list with det=True (it calls exit).
    Thus detection is called once per text ROI with det=True,rec=False; all
    detected line images are then recognized in one det=False,rec=True call.
    The same already initialized PP model is reused and no GPU/model is loaded.
    ``run_ocr_inference`` can be the runtime's shared-model lock wrapper.

    Returned anchor bboxes are exact OCR *line* bboxes in source page pixels and
    units, not invented glyph positions.  Wide A/B lines therefore remain wide.
    They can be inspected to recover a prefix just outside a PP text bbox; this
    function exposes that evidence without enlarging unvalidated original blocks.
    No OCR body is saved or supplied to the VLM.  Failures leave original blocks.
    """
    if len(images_pil_list) != len(high_vlm_blocks):
        raise ValueError("images and high VLM blocks page counts differ")
    if probe_margin_px < 0 or not 0 <= min_anchor_score <= 1:
        raise ValueError("invalid OCR probe margin or anchor confidence")
    if max_lines_per_roi is not None and max_lines_per_roi <= 0:
        raise ValueError("max_lines_per_roi must be positive or None")
    import numpy as np
    if line_cropper is None:
        from mineru.model.ocr.image import get_rotate_crop_image_for_text_rec
        line_cropper = get_rotate_crop_image_for_text_rec
    invoke = run_ocr_inference or (lambda fn, *args, **kwargs: fn(*args, **kwargs))
    started = time.perf_counter()
    anchor_pages = [[{"question": None, "options": [], "ambiguous": False, "anchors": [],
                     "line_boxes_px": [], "line_boxes_unit": []}
                     for _ in page] for page in high_vlm_blocks]
    errors = []
    line_images = []
    line_records = []
    detect_seconds = 0.0
    recognition_seconds = 0.0
    roi_count = 0
    pixels_examined = 0
    for page_index, (image, blocks) in enumerate(zip(images_pil_list, high_vlm_blocks)):
        width, height = image.size
        for block_index, block in enumerate(blocks):
            if block.get("type", block.get("label")) not in _TEXT_TYPES or block.get("angle", 0) not in {None, 0}:
                continue
            unit_box = _bbox(block["bbox"])
            roi = (max(0, math.floor(unit_box[0] * width - probe_margin_px)),
                   max(0, math.floor(unit_box[1] * height - probe_margin_px)),
                   min(width, math.ceil(unit_box[2] * width + probe_margin_px)),
                   min(height, math.ceil(unit_box[3] * height + probe_margin_px)))
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                continue
            meta = anchor_pages[page_index][block_index]
            meta["roi_bbox_px"] = list(roi)
            crop = np.asarray(image.crop(roi).convert("RGB"))[:, :, ::-1].copy()
            roi_count += 1
            pixels_examined += crop.shape[0] * crop.shape[1]
            detect_start = time.perf_counter()
            try:
                result = invoke(ocr_model.ocr, crop, det=True, rec=False, tqdm_enable=False)
                if not isinstance(result, (list, tuple)) or len(result) != 1:
                    raise ValueError("unexpected PP detector result (expected one image)")
                polygons = result[0] if result[0] is not None else []
                for line_index, polygon in enumerate(polygons):
                    points = np.asarray(polygon, dtype=np.float32)
                    if points.shape != (4, 2) or not np.isfinite(points).all():
                        raise ValueError("invalid OCR quadrilateral")
                    points[:, 0] = np.clip(points[:, 0], 0, crop.shape[1])
                    points[:, 1] = np.clip(points[:, 1], 0, crop.shape[0])
                    if points[:, 0].max() <= points[:, 0].min() or points[:, 1].max() <= points[:, 1].min():
                        continue
                    box = [float(points[:, 0].min() + roi[0]), float(points[:, 1].min() + roi[1]),
                           float(points[:, 0].max() + roi[0]), float(points[:, 1].max() + roi[1])]
                    unit_line = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
                    meta["line_boxes_px"].append(box)
                    meta["line_boxes_unit"].append(unit_line)
                    if max_lines_per_roi is not None and line_index >= max_lines_per_roi:
                        continue
                    line_image = line_cropper(crop, points.copy())
                    if line_image is None or not line_image.size:
                        continue
                    line_images.append(line_image)
                    line_records.append({"page": page_index, "block": block_index, "line": line_index,
                                         "bbox_px": box, "bbox_unit": unit_line})
            except Exception as error:
                meta["ambiguous"] = True
                errors.append({"stage": "anchor_detection", "page": page_index, "block": block_index,
                               "error": f"{type(error).__name__}: {error}"})
            finally:
                detect_seconds += time.perf_counter() - detect_start
    recognized = []
    if line_images:
        recognition_start = time.perf_counter()
        try:
            result = invoke(ocr_model.ocr, line_images, det=False, rec=True, tqdm_enable=False)
            if not isinstance(result, (list, tuple)) or len(result) != 1 or result[0] is None:
                raise ValueError("unexpected PP recognition result (expected one batch)")
            recognized = result[0]
            if len(recognized) != len(line_records):
                raise ValueError("PP OCR line recognition count mismatch")
        except Exception as error:
            errors.append({"stage": "anchor_recognition", "error": f"{type(error).__name__}: {error}"})
            recognized = []
            for page in anchor_pages:
                for meta in page:
                    meta["ambiguous"] = True
        finally:
            recognition_seconds = time.perf_counter() - recognition_start
    for record, recognition in zip(line_records, recognized):
        text, score = recognition
        score = float(score)
        if not math.isfinite(score) or score < min_anchor_score:
            continue
        meta = anchor_pages[record["page"]][record["block"]]
        question_match = _QUESTION_START.match(str(text))
        new_anchors = []
        if question_match and int(question_match.group(1)) > 0:
            value = int(question_match.group(1))
            if meta["question"] is None:
                meta["question"] = value
            elif meta["question"] != value:
                meta["ambiguous"] = True
            new_anchors.append(("question", value))
        # A detector can merge adjacent option starts into one wide line.  Keep
        # their order but do not invent separate character boxes for A and B.
        for label in _OPTION.findall(str(text)):
            meta["options"].append(label)
            new_anchors.append(("option", label))
        for kind, value in new_anchors:
            meta["anchors"].append({"kind": kind, "value": value, "score": score,
                                    "bbox_kind": "ocr_line", "line_index": record["line"],
                                    "bbox_px": record["bbox_px"], "bbox_unit": record["bbox_unit"]})
    annotated = [attach_anchor_evidence(blocks, prefixes) for blocks, prefixes in zip(high_vlm_blocks, anchor_pages)]
    merged = [merge_question_blocks(blocks, prefixes, image.size, margin_px=margin_px)
              for blocks, prefixes, image in zip(annotated, anchor_pages, images_pil_list)]
    timings = {"seconds": round(time.perf_counter() - started, 6),
               "detection_seconds": round(detect_seconds, 6),
               "recognition_seconds": round(recognition_seconds, 6),
               "text_rois": roi_count, "recognized_lines": len(recognized),
               "roi_pixels": pixels_examined,
               "anchor_count": sum(len(meta["anchors"]) for page in anchor_pages for meta in page),
               "merged_questions": sum(len(result.groups) for result in merged)}
    return PageMergeResult([result.blocks for result in merged], anchor_pages,
                           [result.groups for result in merged], [result.rejected for result in merged],
                           tuple(errors), timings)


def _ocr_self_test():
    import numpy as np
    from PIL import Image
    blocks = [{"type": "paragraph_title", "bbox": [0.02, 0.005, 0.98, 0.03]},
              {"type": "text", "bbox": [0.02, 0.05, 0.7, 0.07]},
              {"type": "text", "bbox": [0.02, 0.10, 0.98, 0.145]},
              {"type": "text", "bbox": [0.02, 0.19, 0.98, 0.235]},
              {"type": "text", "bbox": [0.02, 0.30, 0.7, 0.325]}]

    class FakeOCR:
        def __init__(self, fail=False):
            self.calls = []
            self.fail = fail

        def ocr(self, image, det, rec, **kwargs):
            self.calls.append((det, rec, isinstance(image, list)))
            if det:
                assert not isinstance(image, list), "native OCR det cannot accept list"
                height, width = image.shape[:2]
                return [[[[2, 2], [width - 2, 2], [width - 2, height - 2], [2, height - 2]]]]
            if self.fail:
                return [[]]
            assert isinstance(image, list) and len(image) == 4
            return [[("(3) private body must not survive", 0.99), ("(A) aaa (B) bbb", 0.99),
                     ("(C) ccc (D) ddd", 0.99), ("(4) following question", 0.99)]]

    def rectangle_crop(image, points):
        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int)
        return image[y0:y1, x0:x1].copy()

    image = Image.new("RGB", (1000, 1000), "white")
    fake = FakeOCR()
    before = deepcopy(blocks)
    result = merge_page_question_regions([image], [blocks], fake, line_cropper=rectangle_crop)
    assert blocks == before and len(result.groups[0]) == 1
    assert result.groups[0][0]["question"] == 3
    assert fake.calls == [(True, False, False)] * 4 + [(False, True, True)]
    assert result.anchor_pages[0][1]["anchors"][0]["bbox_px"] == [14.0, 44.0, 706.0, 76.0]
    assert "private body" not in repr(result)
    assert result.timings["recognized_lines"] == 4 and result.timings["anchor_count"] == 6
    failed = merge_page_question_regions([image], [blocks], FakeOCR(fail=True), line_cropper=rectangle_crop)
    assert all(b["bbox"] == source["bbox"] and b["type"] == source["type"]
               for b, source in zip(failed.blocks_list[0], blocks)) and len(failed.errors) == 1
    assert not failed.groups[0]
    combined = result.blocks_list[0][1]
    assert combined["_converter_source_indices"] == [1, 2, 3]
    assert combined["_converter_anchor_evidence"]["options"] == list("ABCD")
    assert len(combined["_converter_anchor_evidence"]["anchors"]) == 5
    assert len(combined["_converter_anchor_evidence"]["line_boxes_px"]) == 3
    assert result.anchor_pages[0][1]["line_boxes_px"] == [[14.0, 44.0, 706.0, 76.0]]
    print("PASS: FakeOCR sequential detectors + one batched recognizer; exact source OCR-line bboxes; anchor-only metadata; malformed OCR fallback")


def _self_test():
    size = (1000, 1000)
    blocks = [{"type": "paragraph_title", "bbox": [20, 5, 980, 30], "content": "title"},
              {"type": "text", "bbox": [20, 50, 700, 70], "content": "OCR stem must not leak"},
              {"type": "text", "bbox": [20, 100, 980, 145]},
              {"type": "inline_formula", "bbox": [20, 150, 300, 175]},
              {"type": "text", "bbox": [20, 190, 980, 235]},
              {"type": "text", "bbox": [20, 300, 700, 325]},
              {"type": "page_number", "bbox": [470, 950, 530, 970]}]
    prefixes = ["(1) heading is protected", "(3)", "(A) (B)", None, "(C) (D)", "(4)", None]
    before = deepcopy(blocks)
    result = merge_question_blocks(blocks, prefixes, size, units="pixel")
    assert blocks == before
    assert len(result.groups) == 1 and result.groups[0]["source_indices"] == [1, 2, 3, 4]
    assert result.groups[0]["bbox_px"] == [16.0, 46.0, 984.0, 239.0]
    assert [b["type"] for b in result.blocks] == ["paragraph_title", "text", "text", "page_number"]
    assert "content" not in result.blocks[1] and "lines" not in result.blocks[1]
    assert result.blocks[0] == blocks[0] and result.blocks[-1] == blocks[-1]
    missing = prefixes.copy(); missing[4] = "(C)"
    assert merge_question_blocks(blocks, missing, size, units="pixel").blocks == blocks
    reversed_questions = prefixes.copy(); reversed_questions[5] = "(2)"
    assert merge_question_blocks(blocks, reversed_questions, size, units="pixel").blocks == blocks
    prose = ["some body"] * len(blocks)
    assert merge_question_blocks(blocks, prose, size, units="pixel").blocks == blocks
    # Independent images and tables must remain outside the joint crop.
    for visual_type in ("image", "table", "header", "paragraph_title"):
        with_visual = blocks[:3] + [{"type": visual_type, "bbox": [400, 145, 500, 180]}] + blocks[3:]
        visual_prefixes = prefixes[:3] + [None] + prefixes[3:]
        assert merge_question_blocks(with_visual, visual_prefixes, size, units="pixel").blocks == with_visual
    column_jump = deepcopy(blocks); column_jump[5]["bbox"] = [550, 20, 980, 50]
    assert merge_question_blocks(column_jump, prefixes, size, units="pixel").blocks == column_jump
    # A formula reaching the next stem cancels grouping rather than clipping it.
    crossing = deepcopy(blocks); crossing[3]["bbox"] = [20, 150, 300, 310]
    assert merge_question_blocks(crossing, prefixes, size, units="pixel").blocks == crossing
    ambiguous = prefixes.copy(); ambiguous[2] = "(A) (B) (5) (6)"
    assert merge_question_blocks(blocks, ambiguous, size, units="pixel").blocks == blocks
    unit_blocks = [{**b, "bbox": [x / 1000 for x in b["bbox"]]} for b in blocks]
    unit = merge_question_blocks(unit_blocks, prefixes, size)
    assert unit.groups == result.groups
    assert len(unit.blocks) == len(result.blocks)
    # Twelve-pixel retries must share blank gutters and never capture neighbours.
    close = [{"type": "text", "bbox": [10, 10, 100, 100]},
             {"type": "text", "bbox": [110, 10, 200, 100]},
             {"type": "image", "bbox": [10, 110, 100, 200]}]
    padding = pad_text_blocks_safely(close, size, units="pixel")
    assert padding[0]["bbox"][2] == padding[1]["bbox"][0] == 105
    assert padding[0]["bbox"][3] == 105 and padding[2] == close[2]
    assert close[0]["bbox"] == [10, 10, 100, 100]
    assert padding[0]["_converter_unpadded_bbox"] == close[0]["bbox"]
    print("PASS: joint cross-row formulas, A/B/C/D order, 4px margins, input immutability, source ownership, no OCR body leakage, titles/prose/images/tables fallback, unit coordinates")


if __name__ == "__main__":
    _self_test()
    import sys
    if "--test-ocr" in sys.argv:
        _ocr_self_test()
