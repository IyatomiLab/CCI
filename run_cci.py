from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cci.cci import GeneralityScorer, compute_cci
from cci.llm_client import LLMClient, LLMConfig
from cci.output_formatter import build_output_row, save_output_csv

# G20 member countries only
GLOBAL_MODE = [
    "Argentina",
    "Australia",
    "Brazil",
    "Canada",
    "China",
    "France",
    "Germany",
    "India",
    "Indonesia",
    "Italy",
    "Japan",
    "Mexico",
    "Republic of Korea",
    "Republic of South Africa",
    "Russian Federation",
    "Saudi Arabia",
    "Turkey",
    "United Kingdom",
    "United States of America",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CCI for a single text input.")

    parser.add_argument("--text", type=str, required=True, help="Input sentence/text")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_cci.csv"),
        help="Output CSV",
    )

    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument(
        "--cultures",
        nargs="*",
        type=str,
        default=GLOBAL_MODE,
        help="Country names to use as the culture set for CCI. Default is G20 members.",
    )
    parser.add_argument("--target-culture", type=str, default="Japan")
    return parser.parse_args()


def build_scorer(args: argparse.Namespace) -> GeneralityScorer:
    llm_cfg = LLMConfig(
        model_id=args.model,
        cache_dir=args.cache_dir or None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    llm = LLMClient(llm_cfg)
    return GeneralityScorer(
        llm,
        n_samples=args.n_samples,
    )


def main() -> None:
    args = parse_args()

    text = args.text.strip()
    calc = build_scorer(args)

    try:
        p, cci = compute_cci(
            x=text,
            cultures=list(args.cultures),
            target_culture=args.target_culture,
            calc=calc,
        )
        print(f"CCI: {cci:.4f} | {text}")
        scores_str = ", ".join(f"'{k}': {v:.3f}" for k, v in p.items())
        print(f"Generality Scores: {{{scores_str}}}")
    except Exception as e:
        logging.exception("Computation failed")
        p = {"error": f"{type(e).__name__}: {e}"}
        cci = ""

    row = build_output_row(
        text=text,
        model_id=args.model,
        target_culture=args.target_culture,
        p=p,
        cci=cci,
    )
    save_output_csv(args.out, [row])


if __name__ == "__main__":
    main()
