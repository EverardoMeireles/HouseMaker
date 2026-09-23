# ### Imports ###
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from housemaker.plan_correction_models import (
    OPENAI_PLAN_CORRECTION_MODELS,
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)

# ### Public constants ###
OPENAI_PRICING_MARKDOWN_URL = (
    "https://developers.openai.com/api/docs/pricing.md"
)
OPENAI_PRICING_RESPONSE_LIMIT_BYTES = 256 * 1024
OPENAI_PRICING_TIMEOUT_MILLISECONDS = 5_000
PRICE_NOT_FOUND_TEXT = "Price not found"
LOCAL_MODEL_PRICE_TEXT = "Local; no API token charge"


# ### Parsing constants ###
_FLAGSHIP_SECTION_START = "Flagship models"
_FLAGSHIP_SECTION_END = "Cyber models"
_STANDARD_PRICING_HEADING = "### Standard pricing data"
_FLAGSHIP_HEADER = (
    "Model",
    "Short context input",
    "Short context cached input",
    "Short context cache writes",
    "Short context output",
    "Long context input",
    "Long context cached input",
    "Long context cache writes",
    "Long context output",
)
_IMAGE_SECTION_START = "Image generation models"
_IMAGE_SECTION_END = "Video generation models"
_STANDARD_TIER_LABEL = "Standard"
_BATCH_TIER_LABEL = "Batch"
_IMAGE_HEADER = (
    "Model",
    "Modality",
    "Input",
    "Cached input",
    "Output",
)
_PRICE_PATTERN = re.compile(r"^\$(?:0|[1-9]\d*)(?:\.\d+)?$")


# ### Public data models ###
@dataclass(frozen=True)
class TokenRate:
    """One Standard API token rate, expressed in USD per million tokens."""

    modality: str
    input_usd: Decimal
    output_usd: Decimal | None


@dataclass(frozen=True)
class ModelTokenPricing:
    """The Standard token rates needed for one plan-correction model."""

    model_id: str
    rates: tuple[TokenRate, ...]


# ### Public parsing API ###
def parse_openai_pricing_markdown(
    markdown: str,
) -> dict[str, ModelTokenPricing]:
    """Parse supported Standard rates from OpenAI's public pricing Markdown.

    The document publishes several processing tiers and repeats model rows.
    Parsing is intentionally strict so a documentation-layout change produces
    ``Price not found`` instead of silently displaying a Batch, Flex, Fast, or
    long-context rate as the normal Standard price.
    """

    if not isinstance(markdown, str) or not markdown.strip():
        return {}
    lines = tuple(line.strip() for line in markdown.splitlines())
    prices: dict[str, ModelTokenPricing] = {}
    prices.update(_parse_flagship_prices(lines))
    image_price = _parse_gpt_image_2_price(lines)
    if image_price is not None:
        prices[image_price.model_id] = image_price
    return prices


# ### Public display helpers ###
def format_plan_correction_model_label(
    base_label: str,
    model_id: str,
    pricing: ModelTokenPricing | None,
) -> str:
    """Return one model name with its current price or exact fallback text."""

    normalized_label = str(base_label).strip() or str(model_id).strip()
    normalized_model_id = str(model_id).strip()
    if normalized_model_id == PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1:
        return f"{normalized_label} ({LOCAL_MODEL_PRICE_TEXT})"
    if normalized_model_id not in OPENAI_PLAN_CORRECTION_MODELS:
        return normalized_label
    if pricing is None or pricing.model_id != normalized_model_id:
        return f"{normalized_label} ({PRICE_NOT_FOUND_TEXT})"

    if normalized_model_id == PLAN_CORRECTION_MODEL_GPT_IMAGE_2:
        text_rate = _rate_for_modality(pricing, "Text")
        image_rate = _rate_for_modality(pricing, "Image")
        if (
            text_rate is None
            or image_rate is None
            or image_rate.output_usd is None
        ):
            return f"{normalized_label} ({PRICE_NOT_FOUND_TEXT})"
        summary = (
            f"{_format_usd(text_rate.input_usd)} text in, "
            f"{_format_usd(image_rate.input_usd)} image in / "
            f"{_format_usd(image_rate.output_usd)} image out per 1M tokens"
        )
        return f"{normalized_label} ({summary})"

    token_rate = _rate_for_modality(pricing, "Tokens")
    if token_rate is None or token_rate.output_usd is None:
        return f"{normalized_label} ({PRICE_NOT_FOUND_TEXT})"
    summary = (
        f"{_format_usd(token_rate.input_usd)} in / "
        f"{_format_usd(token_rate.output_usd)} out per 1M tokens"
    )
    return f"{normalized_label} ({summary})"


# ### Flagship pricing parser ###
def _parse_flagship_prices(
    lines: tuple[str, ...],
) -> dict[str, ModelTokenPricing]:
    section = _slice_between_unique_labels(
        lines,
        _FLAGSHIP_SECTION_START,
        _FLAGSHIP_SECTION_END,
    )
    if section is None:
        return {}
    table = _table_after_unique_heading(section, _STANDARD_PRICING_HEADING)
    if table is None or not table or table[0] != _FLAGSHIP_HEADER:
        return {}

    rows = table[1:]
    prices: dict[str, ModelTokenPricing] = {}
    for model_id in (
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    ):
        row = _find_unique_row(rows, model_id)
        if row is None or len(row) != len(_FLAGSHIP_HEADER):
            continue
        try:
            rate = TokenRate(
                modality="Tokens",
                input_usd=_parse_usd(row[1]),
                output_usd=_parse_optional_usd(row[4]),
            )
        except ValueError:
            continue
        if rate.output_usd is None:
            continue
        prices[model_id] = ModelTokenPricing(model_id, (rate,))
    return prices


# ### Image pricing parser ###
def _parse_gpt_image_2_price(
    lines: tuple[str, ...],
) -> ModelTokenPricing | None:
    section = _slice_between_unique_labels(
        lines,
        _IMAGE_SECTION_START,
        _IMAGE_SECTION_END,
    )
    if section is None:
        return None
    standard_section = _slice_between_unique_labels(
        section,
        _STANDARD_TIER_LABEL,
        _BATCH_TIER_LABEL,
    )
    if standard_section is None:
        return None
    table = _first_markdown_table(standard_section)
    if table is None or not table or table[0] != _IMAGE_HEADER:
        return None

    rows = tuple(
        row
        for row in table[1:]
        if row and row[0] == PLAN_CORRECTION_MODEL_GPT_IMAGE_2
    )
    text_row = _find_unique_row(rows, "Text", column=1)
    image_row = _find_unique_row(rows, "Image", column=1)
    if text_row is None or image_row is None:
        return None
    try:
        text_rate = _parse_modality_rate(text_row)
        image_rate = _parse_modality_rate(image_row)
    except ValueError:
        return None
    if image_rate.output_usd is None:
        return None
    return ModelTokenPricing(
        PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        (text_rate, image_rate),
    )


def _parse_modality_rate(row: tuple[str, ...]) -> TokenRate:
    if len(row) != len(_IMAGE_HEADER):
        raise ValueError("Unexpected pricing row width.")
    return TokenRate(
        modality=row[1],
        input_usd=_parse_usd(row[2]),
        output_usd=_parse_optional_usd(row[4]),
    )


# ### Markdown table helpers ###
def _slice_between_unique_labels(
    lines: tuple[str, ...],
    start_label: str,
    end_label: str,
) -> tuple[str, ...] | None:
    start_indexes = tuple(
        index for index, line in enumerate(lines) if line == start_label
    )
    if len(start_indexes) != 1:
        return None
    start_index = start_indexes[0]
    end_indexes = tuple(
        index
        for index, line in enumerate(lines)
        if index > start_index and line == end_label
    )
    if len(end_indexes) != 1:
        return None
    return lines[start_index + 1 : end_indexes[0]]


def _table_after_unique_heading(
    lines: tuple[str, ...],
    heading: str,
) -> tuple[tuple[str, ...], ...] | None:
    indexes = tuple(index for index, line in enumerate(lines) if line == heading)
    if len(indexes) != 1:
        return None
    return _first_markdown_table(lines[indexes[0] + 1 :])


def _first_markdown_table(
    lines: tuple[str, ...],
) -> tuple[tuple[str, ...], ...] | None:
    start_index = next(
        (index for index, line in enumerate(lines) if line.startswith("|")),
        None,
    )
    if start_index is None:
        return None
    raw_rows: list[tuple[str, ...]] = []
    for line in lines[start_index:]:
        if not line.startswith("|"):
            break
        parsed = _parse_markdown_row(line)
        if parsed is None:
            return None
        raw_rows.append(parsed)
    if len(raw_rows) < 3 or not _is_separator_row(raw_rows[1]):
        return None
    return (raw_rows[0], *raw_rows[2:])


def _parse_markdown_row(line: str) -> tuple[str, ...] | None:
    if not line.startswith("|") or not line.endswith("|"):
        return None
    cells = tuple(cell.strip() for cell in line[1:-1].split("|"))
    return cells if cells and all(cells) else None


def _is_separator_row(row: tuple[str, ...]) -> bool:
    return bool(row) and all(
        len(cell) >= 3 and set(cell) <= {"-", ":"} for cell in row
    )


def _find_unique_row(
    rows: tuple[tuple[str, ...], ...],
    value: str,
    *,
    column: int = 0,
) -> tuple[str, ...] | None:
    matches = tuple(
        row for row in rows if len(row) > column and row[column] == value
    )
    return matches[0] if len(matches) == 1 else None


# ### Price value helpers ###
def _parse_usd(value: str) -> Decimal:
    if not _PRICE_PATTERN.fullmatch(value):
        raise ValueError("Invalid USD token price.")
    try:
        parsed = Decimal(value[1:])
    except InvalidOperation:
        raise ValueError("Invalid USD token price.") from None
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("Invalid USD token price.")
    return parsed


def _parse_optional_usd(value: str) -> Decimal | None:
    return None if value == "-" else _parse_usd(value)


def _rate_for_modality(
    pricing: ModelTokenPricing,
    modality: str,
) -> TokenRate | None:
    matches = tuple(rate for rate in pricing.rates if rate.modality == modality)
    return matches[0] if len(matches) == 1 else None


def _format_usd(value: Decimal) -> str:
    integral, _, fractional = format(value, "f").partition(".")
    displayed_fraction = fractional.rstrip("0")
    if len(displayed_fraction) < 2:
        displayed_fraction = displayed_fraction.ljust(2, "0")
    return f"${integral}.{displayed_fraction}"
