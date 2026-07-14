"""Shared case formatting utilities for the uveitis Qwen-VL pipeline."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


LABELS = ("感染性葡萄膜炎", "非感染性葡萄膜炎")

CLINICAL_COLUMNS = [
    "白细胞计数WBC",
    "中性粒细胞百分比NEU%",
    "淋巴细胞百分比LYM%",
    "单核细胞百分比MON%",
    "嗜酸性粒细胞百分比EOS%",
    "嗜碱性粒细胞百分比BAS%",
    "中性粒细胞数NEU#",
    "淋巴细胞数LYM#",
    "单核细胞数MON#",
    "嗜酸性粒细胞数EOS#",
    "嗜碱性粒细胞数BAS#",
    "红细胞数RBC",
    "血红蛋白HGB",
    "红细胞比容HCT",
    "平均红细胞体积MCV",
    "平均红细胞血红蛋白含量MCH",
    "平均红细胞血红蛋白浓度MCHC",
    "红细胞分布宽度变异系数RDW-CV",
    "红细胞分布宽度标准差RDW-SD",
    "血小板计数PLT",
    "平均血小板体积MPV",
    "血小板分布宽度PDW",
    "血小板压积PCT",
]


CLINICAL_ALIASES = {
    "白细胞计数WBC": ["白细胞计数WBC", "白细胞计数", "白细胞总数", "白细胞", "白血球", "WBC"],
    "中性粒细胞百分比NEU%": [
        "中性粒细胞百分比NEU%",
        "中性粒细胞百分比",
        "中性粒细胞百分数",
        "中性粒细胞比例",
        "中性粒细胞比率",
        "中性粒百分比",
        "NEU%",
        "NEUT%",
        "NE%",
        "GRAN%",
        "GR%",
    ],
    "淋巴细胞百分比LYM%": [
        "淋巴细胞百分比LYM%",
        "淋巴细胞百分比",
        "淋巴细胞百分数",
        "淋巴细胞比例",
        "淋巴细胞比率",
        "LYM%",
        "LYMPH%",
        "LY%",
        "L%",
    ],
    "单核细胞百分比MON%": [
        "单核细胞百分比MON%",
        "单核细胞百分比",
        "单核细胞百分数",
        "单核细胞比例",
        "单核细胞比率",
        "MON%",
        "MONO%",
    ],
    "嗜酸性粒细胞百分比EOS%": [
        "嗜酸性粒细胞百分比EOS%",
        "嗜酸性粒细胞百分比",
        "嗜酸性粒细胞百分数",
        "嗜酸性粒细胞比例",
        "嗜酸性粒细胞比率",
        "EOS%",
        "EOSIN%",
        "EO%",
        "E%",
    ],
    "嗜碱性粒细胞百分比BAS%": [
        "嗜碱性粒细胞百分比BAS%",
        "嗜碱性粒细胞百分比",
        "嗜碱性粒细胞百分数",
        "嗜碱性粒细胞比例",
        "嗜碱性粒细胞比率",
        "BAS%",
        "BASO%",
        "BA%",
    ],
    "中性粒细胞数NEU#": [
        "中性粒细胞数NEU#",
        "中性粒细胞数",
        "中性粒细胞绝对值",
        "NEU#",
        "NEUT#",
        "NE#",
        "GRAN#",
        "GR#",
        "ANC",
    ],
    "淋巴细胞数LYM#": [
        "淋巴细胞数LYM#",
        "淋巴细胞数",
        "淋巴细胞绝对值",
        "LYM#",
        "LYMPH#",
        "LY#",
        "L#",
        "ALC",
    ],
    "单核细胞数MON#": ["单核细胞数MON#", "单核细胞数", "单核细胞绝对值", "MON#", "MONO#"],
    "嗜酸性粒细胞数EOS#": ["嗜酸性粒细胞数EOS#", "嗜酸性粒细胞数", "嗜酸性粒细胞绝对值", "EOS#", "EOSIN#", "EO#", "E#"],
    "嗜碱性粒细胞数BAS#": ["嗜碱性粒细胞数BAS#", "嗜碱性粒细胞数", "嗜碱性粒细胞绝对值", "BAS#", "BASO#", "BA#"],
    "红细胞数RBC": ["红细胞数RBC", "红细胞数", "红细胞", "RBC"],
    "血红蛋白HGB": ["血红蛋白HGB", "血红蛋白", "HGB", "Hb", "HB"],
    "红细胞比容HCT": ["红细胞比容HCT", "红细胞比容", "红细胞压积", "HCT", "PCV"],
    "平均红细胞体积MCV": ["平均红细胞体积MCV", "平均红细胞体积", "MCV"],
    "平均红细胞血红蛋白含量MCH": ["平均红细胞血红蛋白含量MCH", "平均红细胞血红蛋白含量", "MCH"],
    "平均红细胞血红蛋白浓度MCHC": ["平均红细胞血红蛋白浓度MCHC", "平均红细胞血红蛋白浓度", "MCHC"],
    "红细胞分布宽度变异系数RDW-CV": ["红细胞分布宽度变异系数RDW-CV", "红细胞分布宽度变异系数", "RDW-CV", "RDWCV"],
    "红细胞分布宽度标准差RDW-SD": ["红细胞分布宽度标准差RDW-SD", "红细胞分布宽度标准差", "RDW-SD", "RDWSD"],
    "血小板计数PLT": ["血小板计数PLT", "血小板计数", "血小板", "PLT"],
    "平均血小板体积MPV": ["平均血小板体积MPV", "平均血小板体积", "MPV"],
    "血小板分布宽度PDW": ["血小板分布宽度PDW", "血小板分布宽度", "PDW"],
    "血小板压积PCT": ["血小板压积PCT", "血小板压积", "血小板比容", "PCT"],
}


CLINICAL_ALIAS_TO_COLUMN = {
    unicodedata.normalize("NFKC", alias).strip().lower(): column
    for column, aliases in CLINICAL_ALIASES.items()
    for alias in aliases
}

NUMBER_PATTERN = r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
NUMBER_RE = re.compile(NUMBER_PATTERN)
RESULT_MARKER_PATTERN = r"(?:[<>≤≥]=?|[↑↓]|[HhLl]|高|低|\*)"
RESULT_VALUE_LABEL_PATTERN = r"(?:检验结果|检测结果|测定结果|结果值|测定值|结果|数值|result|value)"
PCT_NON_BLOOD_ROUTINE_TERMS = ("降钙素原", "前降钙素", "procalcitonin", "pro-calcitonin")
PCT_BLOOD_ROUTINE_CONTEXT_TERMS = ("血小板压积", "血小板比容", "plateletcrit", "platelet crit")
PCT_NON_BLOOD_ROUTINE_UNIT_RE = re.compile(r"^\s*(?:ng|pg|ug|µg)\s*/\s*(?:ml|l)\b", flags=re.IGNORECASE)
LAB_SEGMENT_DELIMITERS = "\n\r;；,，"


def normalize_lab_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def clean_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = normalize_lab_text(value)
    if not text or text.lower() == "nan":
        return None
    return text


def parse_numeric_value(value: Any) -> float | None:
    text = clean_value(value)
    if text is None:
        return None
    match = NUMBER_RE.search(text)
    if match is None:
        return None
    raw = match.group(0).replace(",", ".")
    if raw.startswith("."):
        raw = "0" + raw
    elif raw.startswith("-."):
        raw = "-0" + raw[1:]
    elif raw.startswith("+."):
        raw = "+0" + raw[1:]
    try:
        return float(raw)
    except ValueError:
        return None


def text_segment_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(delimiter, 0, start) for delimiter in LAB_SEGMENT_DELIMITERS)
    right_candidates = [
        index
        for delimiter in LAB_SEGMENT_DELIMITERS
        if (index := text.find(delimiter, end)) != -1
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right]


def is_non_blood_routine_pct_context(text: str, match: re.Match[str]) -> bool:
    segment = text_segment_around(text, match.start(), match.end()).lower()
    if any(term.lower() in segment for term in PCT_BLOOD_ROUTINE_CONTEXT_TERMS):
        return False
    if any(term.lower() in segment for term in PCT_NON_BLOOD_ROUTINE_TERMS):
        return True
    after_number = text[match.end(1) : match.end(1) + 24]
    return bool(PCT_NON_BLOOD_ROUTINE_UNIT_RE.match(after_number))


def sex_text(value: Any) -> str:
    text = clean_value(value)
    if text == "1":
        return "男"
    if text == "2":
        return "女"
    return text or "未知"


def format_number(value: Any) -> str | None:
    text = clean_value(value)
    if text is None:
        return None
    number = parse_numeric_value(text)
    if number is None:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.4g}"


def normalize_labs_mapping(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        raw_key = normalize_lab_text(key)
        column = CLINICAL_ALIAS_TO_COLUMN.get(raw_key.lower())
        if column is None:
            continue
        if column in normalized and raw_key != column:
            continue
        normalized[column] = value
    return normalized


def lab_items_from_mapping(values: dict[str, Any]) -> list[tuple[str, str]]:
    normalized = normalize_labs_mapping(values)
    items: list[tuple[str, str]] = []
    for column in CLINICAL_COLUMNS:
        value = format_number(normalized.get(column))
        if value is not None:
            items.append((column, value))
    return items


def labs_from_prompt(prompt: str) -> dict[str, float]:
    values: dict[str, float] = {}
    normalized_prompt = normalize_lab_text(prompt)
    boundary = r"(?<![A-Za-z0-9#%\u4e00-\u9fff])"
    optional_parenthetical = r"(?:\s*\([^)\n]{0,32}\))?"
    result_markers = rf"(?:\s*{RESULT_MARKER_PATTERN}\s*)*"
    separator = rf"(?:\s*(?:[:=\-]\s*)|\s+|(?={result_markers}{NUMBER_PATTERN}))"
    optional_result_label = rf"(?:\s*{RESULT_VALUE_LABEL_PATTERN}\s*(?:[:=\-]\s*)?)?"
    for column in CLINICAL_COLUMNS:
        aliases = sorted({normalize_lab_text(alias) for alias in CLINICAL_ALIASES[column]}, key=len, reverse=True)
        for alias in aliases:
            pattern = re.compile(
                (
                    rf"{boundary}{re.escape(alias)}{optional_parenthetical}{result_markers}"
                    rf"{separator}{optional_result_label}{result_markers}({NUMBER_PATTERN})"
                ),
                flags=re.MULTILINE | re.IGNORECASE,
            )
            match = pattern.search(normalized_prompt)
            if match is None:
                continue
            if (
                column == "血小板压积PCT"
                and normalize_lab_text(alias).lower() == "pct"
                and is_non_blood_routine_pct_context(normalized_prompt, match)
            ):
                continue
            number = parse_numeric_value(match.group(1))
            if number is not None:
                values[column] = number
                break
    return values


def lab_feature_vector(labs: dict[str, Any]) -> list[float | None]:
    normalized = normalize_labs_mapping(labs)
    features: list[float | None] = []
    for column in CLINICAL_COLUMNS:
        features.append(parse_numeric_value(normalized.get(column)))
    return features


def lab_items_from_row(row: Any) -> list[tuple[str, str]]:
    return lab_items_from_mapping({column: row.get(column) for column in CLINICAL_COLUMNS})


def build_prompt(
    sex: Any,
    age: Any,
    chief_complaint: Any,
    labs: dict[str, Any],
) -> str:
    parts = [
        "请根据欧堡眼底影像、主诉和血常规指标，判断该葡萄膜炎病例更符合哪一类。",
        "只能回答以下两个标签之一：感染性葡萄膜炎、非感染性葡萄膜炎。",
        "",
        f"性别：{sex_text(sex)}",
        f"年龄：{clean_value(age) or '未知'}",
        f"主诉：{clean_value(chief_complaint) or '未提供'}",
        "",
        "血常规指标：",
    ]
    lab_items = lab_items_from_mapping(labs)
    if lab_items:
        parts.extend(f"- {name}：{value}" for name, value in lab_items)
    else:
        parts.append("- 未提供")
    return "\n".join(parts)


def build_image_only_prompt() -> str:
    return "\n".join(
        [
            "请根据欧堡眼底影像，判断该葡萄膜炎病例更符合哪一类。",
            "只能回答以下两个标签之一：感染性葡萄膜炎、非感染性葡萄膜炎。",
        ]
    )


def build_prompt_from_row(row: Any) -> str:
    labs = {column: row.get(column) for column in CLINICAL_COLUMNS}
    return build_prompt(
        sex=row.get("性别"),
        age=row.get("年龄"),
        chief_complaint=row.get("主诉"),
        labs=labs,
    )


def prompt_has_lab_values(prompt: str) -> bool:
    return bool(labs_from_prompt(prompt))


def normalize_prediction(text: str) -> str | None:
    cleaned = text.strip()
    if cleaned in LABELS:
        return cleaned

    if "非感染性葡萄膜炎" in cleaned:
        remainder = cleaned.replace("非感染性葡萄膜炎", "")
        if "感染性葡萄膜炎" not in remainder:
            return "非感染性葡萄膜炎"
        return None
    if "感染性葡萄膜炎" in cleaned:
        return "感染性葡萄膜炎"
    return None
