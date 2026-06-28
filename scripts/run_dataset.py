"""End-to-end pipeline: data preparation, splitting, and full model evaluation.

Downloads raw datasets (MovieLens, Yelp, Amazon, MIND), processes them into a
standard interaction + item format, builds temporal train/valid/test splits with
candidate pools, and runs all baseline, graph-aware, cross-encoder, and LLM
reranking experiments.

Usage:
    python scripts/run_dataset.py --dataset ml-1m --top_k 200 --seed 42 \
        --config configs/local.yaml

GPU required for LLM reranking steps; can be bypassed with --split_only
(to prepare data on CPU before submitting GPU sweeps).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from coldtail.config import load_config
from coldtail.data.split import create_splits, sample_rerank_users
from coldtail.experiments.failure_analysis import run_failure_analysis
from coldtail.experiments.report import make_report
from coldtail.experiments.retrieval_coverage import run_retrieval_coverage
from coldtail.experiments.run_baselines import run_all_baselines
from coldtail.experiments.run_cross_encoder import run_cross_encoder_rerank
from coldtail.experiments.run_graph_aware import run_graph_aware
from coldtail.experiments.run_llm_rerank import run_llm_rerank
from coldtail.experiments.timing import TimingCollector
from coldtail.utils import seed_everything

RUN_LLM = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_movielens(data_dir: Path, dataset: str):
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "processed" / dataset

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / f"{dataset}.zip"
    target_dir = raw_dir / dataset

    if not zip_path.exists():
        urls = {
            "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
            "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
        }
        if dataset not in urls:
            raise ValueError(f"Unsupported MovieLens variant: {dataset}")
        logger.info(f"Downloading {urls[dataset]}...")
        urllib.request.urlretrieve(urls[dataset], zip_path)

    if not target_dir.exists() and zip_path.exists():
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    logger.info(f"[Data] processing {dataset}...")
    if dataset == "ml-100k":
        ratings = pd.read_csv(
            target_dir / "u.data",
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )
        item_cols = [
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        items = pd.read_csv(
            target_dir / "u.item", sep="|", names=item_cols, encoding="latin-1"
        )
        genre_cols = item_cols[5:]
        items["genres"] = items.apply(
            lambda r: "|".join([g for g in genre_cols if int(r[g]) == 1]) or "unknown",
            axis=1,
        )
        items = items[["item_id", "title", "genres"]]

    elif dataset == "ml-1m":
        ratings = pd.read_csv(
            target_dir / "ratings.dat",
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
        items = pd.read_csv(
            target_dir / "movies.dat",
            sep="::",
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

    elif dataset in ["ml-20m", "ml-25m", "ml-32m"]:
        ratings = pd.read_csv(target_dir / "ratings.csv")
        ratings.rename(
            columns={"userId": "user_id", "movieId": "item_id"}, inplace=True
        )
        items = pd.read_csv(target_dir / "movies.csv")
        items.rename(columns={"movieId": "item_id"}, inplace=True)

    items["text"] = (
        items["title"].fillna("") + " " + items["genres"].fillna("")
    ).str.strip()
    ratings.to_csv(out_dir / "interactions.csv", index=False)
    items.to_csv(out_dir / "items.csv", index=False)
    logger.info(f"[Data] MovieLens done | path={out_dir}")


def prepare_yelp(data_dir: Path, dataset: str):
    raw_yelp_dir = data_dir / "raw" / "yelp"
    if not raw_yelp_dir.exists():
        raise FileNotFoundError(
            f"CRITICAL: {raw_yelp_dir} missing! Please download Yelp JSONs manually."
        )

    out_dir = data_dir / "processed" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse format yelp-{city}-{category}
    parts = dataset.split("-")
    target_city = (
        "Philadelphia"
        if len(parts) > 1 and parts[1].lower() == "philly"
        else parts[1].title()
    )
    target_category = parts[2].title() if len(parts) > 2 else "Restaurants"

    logger.info(
        f"[Data] filtering Yelp | city={target_city} | category={target_category}"
    )

    valid_businesses = {}
    with open(
        raw_yelp_dir / "yelp_academic_dataset_business.json", encoding="utf-8"
    ) as f:
        for line in f:
            b = json.loads(line)
            if (
                b.get("city", "") == target_city
                and b.get("categories")
                and target_category in b.get("categories")
            ):
                valid_businesses[b["business_id"]] = {
                    "item_id": b["business_id"],
                    "title": b.get("name", ""),
                    "genres": b.get("categories").replace(", ", "|"),
                    "text": f"{b.get('name', '')} - {b.get('categories').replace(', ', '|')}",
                }

    valid_b_ids = set(valid_businesses.keys())
    pd.DataFrame.from_dict(valid_businesses, orient="index").to_csv(
        out_dir / "items.csv", index=False
    )

    reviews = []
    with open(
        raw_yelp_dir / "yelp_academic_dataset_review.json", encoding="utf-8"
    ) as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] in valid_b_ids:
                reviews.append(
                    {
                        "user_id": r["user_id"],
                        "item_id": r["business_id"],
                        "rating": r["stars"],
                        "timestamp": pd.to_datetime(r["date"]).timestamp(),
                    }
                )

    pd.DataFrame(reviews).to_csv(out_dir / "interactions.csv", index=False)
    logger.info(f"[Data] Yelp done | path={out_dir}")


def _load_amazon_meta(raw_amazon_dir: Path, category: str) -> pd.DataFrame:
    """Load Amazon-Reviews-2023 item metadata in any of its on-disk formats.

    Two layouts occur in the McAuley-Lab mirror:
      * raw_meta_<category>/full-*.parquet  (or full/data-*.arrow)
        — e.g. Video_Games, Arts_Crafts_and_Sewing
      * meta_<category>.jsonl.gz  at the dataset root
        — e.g. Books (no raw_meta_Books/ dir is provided)
    Returns a frame carrying at least parent_asin, title, categories.
    """
    meta_dir = raw_amazon_dir / f"raw_meta_{category}"

    parquets = sorted(meta_dir.glob("full-*.parquet")) if meta_dir.is_dir() else []
    if parquets:
        logger.info(
            f"[amazon meta] loading {len(parquets)} parquet file(s) from {meta_dir}"
        )
        return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    arrow_dir = meta_dir / "full"
    if arrow_dir.is_dir() and list(arrow_dir.glob("data-*.arrow")):
        arrow_files = sorted(arrow_dir.glob("data-*.arrow"))
        logger.info(
            f"[amazon meta] loading {len(arrow_files)} arrow shard(s) from {arrow_dir}"
        )
        from datasets import load_from_disk

        return load_from_disk(str(arrow_dir)).to_pandas()

    # Gzipped JSONL metadata (e.g. meta_Books.jsonl.gz): stream in chunks and keep only
    # the columns we need — the raw meta carries heavy image/description/video fields.
    gz_meta = raw_amazon_dir / f"meta_{category}.jsonl.gz"
    if gz_meta.exists():
        logger.info(
            f"[amazon meta] loading gzipped JSONL metadata from {gz_meta} (chunked)"
        )
        keep = ["parent_asin", "title", "categories"]
        parts = []
        for chunk in pd.read_json(
            gz_meta, lines=True, compression="gzip", chunksize=200_000
        ):
            for c in keep:
                if c not in chunk.columns:
                    chunk[c] = None
            parts.append(chunk[keep])
        return (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)
        )

    raise FileNotFoundError(
        f"No recognisable metadata for Amazon category '{category}'. Expected one of:\n"
        f"  {meta_dir}/full-*.parquet\n"
        f"  {meta_dir}/full/data-*.arrow\n"
        f"  {gz_meta}\n"
        f"Check that the dataset downloaded successfully."
    )


def prepare_amazon(data_dir: Path, dataset: str):
    import pandas as pd

    out_dir = data_dir / "processed" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_amazon_dir = data_dir / "raw" / "amazon_2023"
    if not raw_amazon_dir.exists():
        raise FileNotFoundError(
            f"CRITICAL: Local Amazon dataset root directory not found at {raw_amazon_dir}.\n"
            f"Please run the 'download_amazon.py' script first to fetch the files."
        )

    cat_query = dataset.replace("amazon-", "").lower().strip()

    local_meta_dirs = [
        d
        for d in raw_amazon_dir.iterdir()
        if d.is_dir() and d.name.startswith("raw_meta_")
    ]

    category_map = {}
    for d in local_meta_dirs:
        clean_key = d.name.replace("raw_meta_", "").lower()
        category_map[clean_key] = d.name.replace("raw_meta_", "")
        category_map[clean_key.replace("_", "")] = d.name.replace("raw_meta_", "")

    category_map.update(
        {
            "beauty": "All_Beauty",
            "toys": "Toys_and_Games",
            "boardgames": "Toys_and_Games",
            "crafts": "Arts_Crafts_and_Sewing",
            "sewing": "Arts_Crafts_and_Sewing",
            "phones": "Cell_Phones_and_Accessories",
            "cellphones": "Cell_Phones_and_Accessories",
            "videogames": "Video_Games",
            "video_games": "Video_Games",
            "movies": "Movies_and_TV",
            "tv": "Movies_and_TV",
            "books": "Books",
            "sports": "Sports_and_Outdoors",
            "sportsandoutdoors": "Sports_and_Outdoors",
            "arts": "Arts_Crafts_and_Sewing",
        }
    )

    category = category_map.get(cat_query)
    if not category:
        available_cats = [d.name.replace("raw_meta_", "") for d in local_meta_dirs]
        raise ValueError(
            f"Unrecognized Amazon category query: '{cat_query}' (Input source: {dataset})\n"
            f"Available categories in local storage: {available_cats}"
        )

    logger.info(f"[Data] loading Amazon metadata | category={category}")
    meta_df = _load_amazon_meta(raw_amazon_dir, category)

    meta_df["text"] = (
        meta_df["title"].fillna("")
        + " - "
        + meta_df["categories"].apply(
            lambda x: (
                "|".join(list(x))
                if hasattr(x, "__iter__") and not isinstance(x, str)
                else str(x)
            )
        )
    )
    items_df = (
        meta_df[["parent_asin", "title", "categories", "text"]]
        .rename(columns={"parent_asin": "item_id", "categories": "genres"})
        .drop_duplicates(subset=["item_id"])
    )

    items_df.to_csv(out_dir / "items.csv", index=False)
    valid_item_ids = set(items_df["item_id"])

    logger.info(f"[Data] loading Amazon reviews (chunked) | category={category}")
    review_file = raw_amazon_dir / "raw" / "review_categories" / f"{category}.jsonl"

    if not review_file.exists():
        raise FileNotFoundError(
            f"Corresponding review JSONL file missing: {review_file}"
        )

    # Stream the review JSONL in chunks: keep only the 4 needed columns and filter to
    # items present in the metadata, per chunk, before concatenating.
    src_cols = ["parent_asin", "user_id", "rating", "timestamp"]
    kept_chunks = []
    n_total = n_kept = 0
    for i, chunk in enumerate(
        pd.read_json(review_file, lines=True, chunksize=1_000_000)
    ):
        n_total += len(chunk)
        chunk = chunk[src_cols]
        chunk = chunk[chunk["parent_asin"].isin(valid_item_ids)]
        if len(chunk):
            kept_chunks.append(chunk)
            n_kept += len(chunk)
        if (i + 1) % 10 == 0:
            logger.info(f"[amazon reviews] scanned {n_total:,} rows, kept {n_kept:,}")

    interactions_df = (
        pd.concat(kept_chunks, ignore_index=True)
        if kept_chunks
        else pd.DataFrame(columns=src_cols)
    ).rename(columns={"parent_asin": "item_id"})

    interactions_df[["user_id", "item_id", "rating", "timestamp"]].to_csv(
        out_dir / "interactions.csv", index=False
    )
    logger.info(
        f"[Data] Amazon reviews done | scanned={n_total:,} kept={n_kept:,} | path={out_dir}"
    )

    logger.info(f"[Data] Amazon done | category={category} | path={out_dir}")


def prepare_mind(data_dir: Path, dataset: str) -> dict[str, set[str]] | None:
    """Prepare MIND dataset; returns impression hard negatives or None."""
    variant = dataset.replace("mind-", "")
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "processed" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"MIND{variant}_train.zip"
    zip_path = raw_dir / zip_name

    if not zip_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: {zip_path.name} missing in {raw_dir}. Please upload the ZIP file."
        )

    target_dir = raw_dir / f"mind-{variant}"
    if not target_dir.exists():
        logger.info(f"Extracting {zip_path.name}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

    news_file = target_dir / "news.tsv"
    behaviors_file = target_dir / "behaviors.tsv"

    if not news_file.exists():
        nested = list(target_dir.rglob("news.tsv"))
        if nested:
            for f in nested[0].parent.iterdir():
                shutil.move(str(f), str(target_dir))
            news_file, behaviors_file = (
                target_dir / "news.tsv",
                target_dir / "behaviors.tsv",
            )
        else:
            raise FileNotFoundError(
                "Extraction failed to find news.tsv inside the ZIP."
            )

    logger.info("[Data] processing MIND items and interactions...")
    news = pd.read_csv(
        news_file,
        sep="\t",
        header=None,
        names=[
            "item_id",
            "category",
            "sub_category",
            "title",
            "abstract",
            "url",
            "entities",
            "entity_abstracts",
        ],
    )
    news["text"] = news["title"].fillna("") + " - " + news["abstract"].fillna("")
    news[["item_id", "category", "title", "text"]].rename(
        columns={"category": "genres"}
    ).to_csv(out_dir / "items.csv", index=False)

    behaviors = pd.read_csv(
        behaviors_file,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )
    behaviors["timestamp"] = pd.to_datetime(behaviors["time"]).astype("int64") // 10**9

    history_rows = behaviors.dropna(subset=["history"]).copy()
    history_rows["item_id"] = history_rows["history"].str.split(" ")
    history_interactions = (
        history_rows[["user_id", "item_id", "timestamp"]]
        .explode("item_id")
        .assign(rating=1)[["user_id", "item_id", "rating", "timestamp"]]
        .reset_index(drop=True)
    )

    impression_rows = behaviors.dropna(subset=["impressions"]).copy()
    impression_rows["impression_token"] = impression_rows["impressions"].str.split(" ")
    all_impressions = impression_rows[
        ["user_id", "impression_token", "timestamp"]
    ].explode("impression_token")
    all_impressions[["item_id", "clicked"]] = all_impressions[
        "impression_token"
    ].str.rsplit("-", n=1, expand=True)

    clicked = all_impressions[all_impressions["clicked"] == "1"]
    impression_interactions = (
        clicked[["user_id", "item_id", "timestamp"]]
        .assign(rating=1)[["user_id", "item_id", "rating", "timestamp"]]
        .reset_index(drop=True)
    )

    not_clicked = all_impressions[all_impressions["clicked"] == "0"]
    hard_neg_df = not_clicked[["user_id", "item_id"]].drop_duplicates()
    impression_hard_negatives: dict[str, set[str]] = {}
    for user, group in hard_neg_df.groupby("user_id"):
        impression_hard_negatives[user] = set(group["item_id"])
    logger.info(
        f"[MIND] extracted impression hard negatives: "
        f"{len(impression_hard_negatives):,} users, "
        f"{hard_neg_df.shape[0]:,} user-item pairs"
    )

    valid_items = set(news["item_id"].astype(str))
    interactions = pd.concat(
        [history_interactions, impression_interactions], ignore_index=True
    )
    interactions = interactions.dropna(subset=["user_id", "item_id"])
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions = interactions[interactions["item_id"].isin(valid_items)]
    interactions = (
        interactions.sort_values("timestamp")
        .drop_duplicates(["user_id", "item_id"], keep="first")
        .reset_index(drop=True)
    )
    interactions.to_csv(out_dir / "interactions.csv", index=False)
    logger.info(f"[Data] MIND done | path={out_dir}")
    return impression_hard_negatives


_SPLIT_ARTIFACTS = [
    "train.csv",
    "valid.csv",
    "test.csv",
    "valid_candidates.csv",
    "candidates.csv",
    "items_mapped.csv",
    "users_mapped.csv",
    "item_stats.csv",
    "rerank_eval_users.csv",  # stratified sample for CrossEncoder / LLM rerankers
]


def splits_exist(processed_dir: Path) -> bool:
    return all((processed_dir / f).exists() for f in _SPLIT_ARTIFACTS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone End-to-End Coldtail Pipeline"
    )
    parser.add_argument(
        "--data_dir", default="data", help="Root directory for data (raw/processed)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name e.g., ml-1m, yelp-philly-restaurants, amazon-beauty, mind-small",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory for logs, models, and reports",
    )
    parser.add_argument(
        "--config", default="configs/local.yaml", help="Path to YAML configuration"
    )
    parser.add_argument(
        "--force_split",
        action="store_true",
        help="Force raw data processing and splitting. If not set, skip if data already exists.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Per-user reranking budget (e.g. 50, 100, 200, 500). "
        "If splitting is required, create_splits builds top_k candidates. "
        "If split artifacts already exist, the same candidate pool is reused "
        "and truncated before scoring. Output directory is suffixed with -top<K>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Overrides config file. Added to output path.",
    )
    parser.add_argument(
        "--max_history",
        type=int,
        default=None,
        help="Truncate user history to last N interactions. "
        "When set, appended to output path as -h<N>.",
    )
    parser.add_argument(
        "--split_only",
        action="store_true",
        help="Only prepare raw data, create the split + candidate pool, and sample "
        "rerank-eval users, then exit before any model evaluation. Use to pre-build "
        "seed-specific splits on CPU before submitting the GPU sweep. The candidate "
        "pool is sized by --top_k, so build with the largest top_k a seed will use.",
    )
    parser.add_argument(
        "--negative_sampling",
        choices=["uniform", "popularity"],
        default="uniform",
        help="How to draw the positive-controlled pool's negatives (sensitivity knob). "
        "'uniform' (default) = unseen items uniformly; 'popularity' = sampled by train "
        "popularity (harder negatives). Non-uniform modes use a SEPARATE split + output "
        "path (s<seed>-neg<mode> / -neg<mode>) so existing results are not clobbered.",
    )
    parser.add_argument(
        "--with_retrieval",
        action="store_true",
        help="Also run the retrieval-realistic coverage experiment (full-catalogue "
        "retrieval, gold NOT injected) after the main pipeline. Writes rank-schema "
        "metrics to the '<out_dir>/retrieval/' subdir so they sit alongside (not on "
        "top of) the positive-controlled rank outputs. Standalone equivalent: "
        "scripts/run_retrieval.py.",
    )
    parser.add_argument(
        "--retrieval_n",
        type=int,
        default=200,
        help="Per-user retrieved pool size for --with_retrieval (default 200, "
        "matching the top-200 rerank budget so coverage = the retrieval ceiling).",
    )
    args = parser.parse_args()

    data_dir_path = Path(args.data_dir).resolve()
    dataset_raw_dir = data_dir_path / "processed" / args.dataset
    # Non-uniform negative sampling uses a separate split + output path so the
    # default (uniform) results are never overwritten.
    neg_suffix = (
        "" if args.negative_sampling == "uniform" else f"-neg{args.negative_sampling}"
    )
    processed_dir_path = dataset_raw_dir / f"s{args.seed}{neg_suffix}"

    out_dir_name = f"{args.dataset}-top{args.top_k}-s{args.seed}"
    if args.max_history is not None:
        out_dir_name += f"-h{args.max_history}"
    out_dir_name += neg_suffix
    out_dir_path = Path(args.output_dir).resolve() / out_dir_name

    logger.info(
        f"[Pipeline] starting | dataset={args.dataset} | top_k={args.top_k} | seed={args.seed}"
    )

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    if args.max_history is not None:
        cfg["max_history"] = args.max_history

    split_cfg = cfg.get("default_split", {}).copy()

    data_family = args.dataset.split("_")[0].split("-")[0].lower()
    specific_cfg = cfg.get("dataset_specific", {}).get(data_family, {})
    if specific_cfg:
        split_cfg.update(specific_cfg)
        logger.info(f"Applied dataset-specific config overrides for '{data_family}'")

    split_cfg["num_negative_candidates"] = args.top_k - 1
    logger.info(
        f"--top_k={args.top_k}: reranking budget. "
        f"If a new split is created, num_negative_candidates={args.top_k - 1}. "
        "If an existing split is reused, candidates are truncated before scoring."
    )

    cfg["split"] = split_cfg
    seed_everything(args.seed)

    s = cfg["split"]
    hard_negatives = None

    if args.force_split or not splits_exist(processed_dir_path):
        if not args.force_split:
            logger.info(
                "[Pipeline] split artifacts not found — running data preparation"
            )
        else:
            logger.info("[Pipeline] --force_split — re-running data preparation")

        logger.info("[Data] preparing raw data...")
        if args.dataset.startswith("ml-"):
            prepare_movielens(data_dir_path, args.dataset)
        elif args.dataset.startswith("yelp-"):
            prepare_yelp(data_dir_path, args.dataset)
        elif args.dataset.startswith("amazon-"):
            prepare_amazon(data_dir_path, args.dataset)
        elif args.dataset.startswith("mind-"):
            mind_hard_neg = prepare_mind(data_dir_path, args.dataset)
            if mind_hard_neg:
                hard_negatives = mind_hard_neg
                logger.info(
                    f"[MIND] will use {len(hard_negatives):,} users' impression hard negatives for candidate pool"
                )
        else:
            raise ValueError(f"Unknown dataset protocol: {args.dataset}")

        logger.info("[Data] preparing splits and candidates...")
        create_splits(
            processed_dir=processed_dir_path,
            raw_dir=dataset_raw_dir,
            rating_threshold=s.get("rating_threshold", 4.0),
            min_user_interactions=s.get("min_user_interactions", 5),
            min_item_interactions=s.get("min_item_interactions", 3),
            valid_per_user=s.get("valid_per_user", 1),
            test_per_user=s.get("test_per_user", 1),
            num_negative_candidates=s.get("num_negative_candidates", 99),
            item_cold_quantile=s.get("item_cold_quantile", 0.05),
            user_cold_quantile=s.get("user_cold_quantile", 0.05),
            long_tail_quantile=s.get("long_tail_quantile", 0.20),
            max_test_users=s.get("max_test_users"),
            train_ratio=s.get("train_ratio", 0.8),
            valid_ratio=s.get("valid_ratio", 0.1),
            seed=args.seed,
            hard_negatives=hard_negatives,
            negative_sampling=args.negative_sampling,
        )

        logger.info("[Data] sampling rerank evaluation users...")
        sample_rerank_users(
            processed_dir=processed_dir_path,
            per_scenario=int(cfg.get("rerank", {}).get("sample_users", 400) / 4),
            seed=args.seed,
        )
    else:
        logger.info("[Pipeline] split artifacts exist — skipping data preparation")

    if args.split_only:
        logger.info(
            f"[Pipeline] --split_only — split artifacts ready at {processed_dir_path}; "
            "skipping all model evaluation"
        )
        return

    rerank_users_path = processed_dir_path / "rerank_eval_users.csv"
    timing = TimingCollector()

    logger.info("[Pipeline] running baselines...")
    run_all_baselines(
        processed_dir_path,
        out_dir_path,
        cfg,
        top_candidates=args.top_k,
        timing_collector=timing,
    )

    logger.info("[Pipeline] running graph-aware fusion...")
    run_graph_aware(processed_dir_path, out_dir_path, cfg, top_candidates=args.top_k)

    logger.info("[Pipeline] running cross-encoder reranking...")
    run_cross_encoder_rerank(
        processed_dir_path,
        out_dir_path,
        cfg,
        top_candidates=args.top_k,
        rerank_users_path=rerank_users_path,
        timing_collector=timing,
    )

    if RUN_LLM:
        logger.info("[Pipeline] running LLM reranking + GALA variants...")
        run_llm_rerank(
            processed_dir_path,
            out_dir_path,
            cfg,
            top_candidates=args.top_k,
            rerank_users_path=rerank_users_path,
            timing_collector=timing,
        )

    timing.save(out_dir_path / "timing_summary.csv")

    logger.info("[Pipeline] running failure analysis...")
    run_failure_analysis(
        processed_dir_path, out_dir_path, cfg, top_candidates=args.top_k
    )

    if args.with_retrieval:
        logger.info("[Pipeline] running retrieval-realistic coverage...")
        run_retrieval_coverage(
            processed_dir=processed_dir_path,
            out_dir=out_dir_path / "retrieval",
            cfg=cfg,
            retrieval_n=args.retrieval_n,
        )

    logger.info("[Pipeline] generating report...")
    report_path = make_report(processed_dir_path, out_dir_path, cfg)

    logger.info(f"[Pipeline] complete | report={report_path}")


if __name__ == "__main__":
    main()
