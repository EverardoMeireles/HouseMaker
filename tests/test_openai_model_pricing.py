# ### Imports ###
from __future__ import annotations

from decimal import Decimal

from housemaker.openai_model_pricing import (
    LOCAL_MODEL_PRICE_TEXT,
    PRICE_NOT_FOUND_TEXT,
    ModelTokenPricing,
    TokenRate,
    format_plan_correction_model_label,
    parse_openai_pricing_markdown,
)
from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)

# ### Markdown fixtures ###
_FLAGSHIP_HEADER = """| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |"""
_IMAGE_HEADER = """| Model | Modality | Input | Cached input | Output |
| --- | --- | --- | --- | --- |"""


def _pricing_markdown() -> str:
    return f"""# Pricing

Flagship models

Standard

### Standard pricing data

{_FLAGSHIP_HEADER}
| gpt-5.6-terra | $2.00 | $0.20 | $2.50 | $12.00 | $4.00 | $0.40 | $5.00 | $18.00 |
| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |

Batch

### Batch pricing data

{_FLAGSHIP_HEADER}
| gpt-5.6-terra | $1.00 | $0.10 | $1.25 | $6.00 | $2.00 | $0.20 | $2.50 | $9.00 |
| gpt-5.6-luna | $0.10 | $0.01 | $0.125 | $0.60 | $0.20 | $0.02 | $0.25 | $0.90 |

Flex

### Flex pricing data

{_FLAGSHIP_HEADER}
| gpt-5.6-terra | $1.00 | $0.10 | $1.25 | $6.00 | $2.00 | $0.20 | $2.50 | $9.00 |
| gpt-5.6-luna | $0.10 | $0.01 | $0.125 | $0.60 | $0.20 | $0.02 | $0.25 | $0.90 |

Fast mode

### Fast pricing data

{_FLAGSHIP_HEADER}
| gpt-5.6-terra | $4.00 | $0.40 | $5.00 | $24.00 | $8.00 | $0.80 | $10.00 | $36.00 |
| gpt-5.6-luna | $0.40 | $0.04 | $0.50 | $2.40 | $0.80 | $0.08 | $1.00 | $3.60 |

Cyber models

Image generation models

Standard

### Grouped Pricing Table data

{_IMAGE_HEADER}
| gpt-image-2 | Image | $8.00 | $2.00 | $30.00 |
| gpt-image-2 | Text | $5.00 | $1.25 | - |

Batch

### Grouped Pricing Table data

{_IMAGE_HEADER}
| gpt-image-2 | Image | $4.00 | $1.00 | $15.00 |
| gpt-image-2 | Text | $2.50 | $0.625 | - |

Video generation models
"""


# ### Parser tests ###
def test_parser_uses_standard_short_context_and_not_discounted_tiers() -> None:
    pricing = parse_openai_pricing_markdown(_pricing_markdown())

    assert format_plan_correction_model_label(
        "GPT-5.6 Luna",
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        pricing[PLAN_CORRECTION_MODEL_GPT_5_6_LUNA],
    ) == "GPT-5.6 Luna ($0.20 in / $1.20 out per 1M tokens)"
    assert format_plan_correction_model_label(
        "GPT-5.6 Terra",
        PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        pricing[PLAN_CORRECTION_MODEL_GPT_5_6_TERRA],
    ) == "GPT-5.6 Terra ($2.00 in / $12.00 out per 1M tokens)"
    assert format_plan_correction_model_label(
        "gpt-image-2",
        PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        pricing[PLAN_CORRECTION_MODEL_GPT_IMAGE_2],
    ) == (
        "gpt-image-2 ($5.00 text in, $8.00 image in / "
        "$30.00 image out per 1M tokens)"
    )


def test_parser_fails_only_the_model_with_an_ambiguous_or_missing_row() -> None:
    markdown = _pricing_markdown().replace(
        "| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |",
        "| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |\n"
        "| gpt-5.6-luna | $9.00 | $9.00 | $9.00 | $9.00 | $9.00 | $9.00 | $9.00 | $9.00 |",
        1,
    )
    pricing = parse_openai_pricing_markdown(markdown)

    assert PLAN_CORRECTION_MODEL_GPT_5_6_LUNA not in pricing
    assert PLAN_CORRECTION_MODEL_GPT_5_6_TERRA in pricing
    assert PLAN_CORRECTION_MODEL_GPT_IMAGE_2 in pricing


def test_parser_rejects_gpt_image_when_only_batch_or_one_modality_exists() -> None:
    markdown = _pricing_markdown().replace(
        "| gpt-image-2 | Text | $5.00 | $1.25 | - |",
        "",
        1,
    )

    pricing = parse_openai_pricing_markdown(markdown)

    assert PLAN_CORRECTION_MODEL_GPT_IMAGE_2 not in pricing
    assert PLAN_CORRECTION_MODEL_GPT_5_6_LUNA in pricing
    assert PLAN_CORRECTION_MODEL_GPT_5_6_TERRA in pricing


def test_parser_returns_no_prices_for_changed_or_invalid_documents() -> None:
    assert parse_openai_pricing_markdown("") == {}
    assert parse_openai_pricing_markdown("not a pricing document") == {}
    assert parse_openai_pricing_markdown(
        _pricing_markdown().replace("$12.00", "USD 12", 1)
    ).keys() == {
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    }


# ### Display fallback tests ###
def test_display_helpers_distinguish_remote_failure_and_local_model() -> None:
    assert format_plan_correction_model_label(
        "GPT-5.6 Luna",
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        None,
    ) == f"GPT-5.6 Luna ({PRICE_NOT_FOUND_TEXT})"
    assert format_plan_correction_model_label(
        "Qwen-Image-2.1",
        PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        None,
    ) == f"Qwen-Image-2.1 ({LOCAL_MODEL_PRICE_TEXT})"


def test_display_helper_preserves_subcent_rates() -> None:
    pricing = ModelTokenPricing(
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        (
            TokenRate(
                modality="Tokens",
                input_usd=Decimal("0.005"),
                output_usd=Decimal("0.015"),
            ),
        ),
    )

    assert format_plan_correction_model_label(
        "GPT-5.6 Luna",
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        pricing,
    ) == "GPT-5.6 Luna ($0.005 in / $0.015 out per 1M tokens)"
