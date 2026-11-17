"""Choose the similarity floor and ceiling for job scoring (run by hand, never by CI or app code).

The score maps a cosine similarity to 0..1 with `clamp((cos - floor) / (ceiling - floor), 0, 1)`.
This script picks `floor` and `ceiling` WITHOUT any relevance judgement:

    floor   = the median (P50) of the cosine similarity of random pairs of stored jobs
              ("no more similar than two jobs picked at random: no signal")
    ceiling = the P99 of the same pairs
              ("closer than 99% of random job pairs")

The pairs come from the extracted jobs of the development database that have a vector made by the
CURRENT embedding model, one job per distinct post text (so the same post stored by two sources is
not compared with itself), sampled with a fixed seed. Nothing is labelled, nothing is scored, and
no CV or relevance data takes part in the choice.

Next to it the script REPORTS (a statement, never used to tune) where the synthetic CV of
`tests/fixtures/cv/alex_example.txt` falls against all those jobs. It makes the CV vector exactly
like an upload does (clean, mask, chunk, embed, normalised mean), so it needs the real model:

    EMBEDDING_PROVIDER=fastembed  (the weights are in the Docker image, see the PR)

If the whole distribution of the CV sits below the floor, the script says so and exits with code 2:
the report is the result, and the defaults must not be re-tuned to hide it.

It reads the database read-only (same guard as the other scripts: only `jobradar`) and writes
nothing. No provider request is made. Only the invented CV is ever used.

    uv run python -m scripts.calibrate_similarity [--pairs 5000] [--seed 1] [--cv PATH | --no-cv]
"""

import argparse
import asyncio
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.cv.text import clean_cv_text, mask_cv_text
from app.matching.embeddings import Embedder, cv_chunks, normalised_mean
from app.matching.factory import build_embedder
from app.models.job import ExtractionStatus, Job
from scripts.db_guard import ScriptError, database_target, ensure_jobs_table, read_only_engine

DEFAULT_CV = Path("tests/fixtures/cv/alex_example.txt")
FLOOR_PERCENTILE = 50.0
CEILING_PERCENTILE = 99.0
TABLE_PERCENTILES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 100.0)
STOP_EXIT_CODE = 2


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (the stored ones have length 1, but do not rely on it)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The q-th percentile (0..100) of a sorted list, with linear interpolation."""
    if not sorted_values:
        raise ValueError("no values")
    position = (len(sorted_values) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def random_pair_cosines(vectors: Sequence[Sequence[float]], pairs: int, seed: int) -> list[float]:
    """Cosines of `pairs` distinct random pairs (all of them when there are fewer pairs)."""
    count = len(vectors)
    total = count * (count - 1) // 2
    rng = random.Random(seed)
    if total <= pairs:
        chosen = [(i, j) for i in range(count) for j in range(i + 1, count)]
    else:
        seen: set[tuple[int, int]] = set()
        while len(seen) < pairs:
            i, j = sorted(rng.sample(range(count), 2))
            seen.add((i, j))
        chosen = sorted(seen)
    return sorted(cosine(vectors[i], vectors[j]) for i, j in chosen)


def table(title: str, sorted_values: Sequence[float]) -> str:
    """A percentile table as text."""
    lines = [
        title,
        f"  n = {len(sorted_values)}   mean = {sum(sorted_values) / len(sorted_values):.4f}",
    ]
    for q in TABLE_PERCENTILES:
        label = "min" if q == 0 else "max" if q == 100 else f"P{q:g}"
        lines.append(f"  {label:>5}  {percentile(sorted_values, q):.4f}")
    return "\n".join(lines)


async def read_job_vectors(settings: Settings) -> list[list[float]]:
    """One vector per distinct post text: extracted jobs with a vector of the current model."""
    engine = read_only_engine(settings.database_url)
    try:
        async with engine.connect() as connection:
            await ensure_jobs_table(connection)
            rows = await connection.execute(
                select(Job.embedding)
                .where(
                    Job.extraction_status == ExtractionStatus.EXTRACTED,
                    Job.embedding.is_not(None),
                    Job.embedding_model == settings.embedding_model_id,
                    Job.raw_text_hash.is_not(None),
                )
                .distinct(Job.raw_text_hash)
                .order_by(Job.raw_text_hash, Job.created_at)
            )
            return [[float(x) for x in row[0]] for row in rows]
    finally:
        await engine.dispose()


def cv_vector(embedder: Embedder, raw_text: str) -> list[float]:
    """The CV vector exactly as an upload makes it: clean, mask, chunk, embed, normalised mean."""
    pieces = cv_chunks(mask_cv_text(clean_cv_text(raw_text)))
    return normalised_mean([embedder.embed([chunk])[0] for chunk in pieces.chunks])


def report_cv(
    cv: Sequence[float], jobs: Sequence[Sequence[float]], floor: float, ceiling: float
) -> tuple[str, bool]:
    """The CV's distribution against all jobs, and whether all of it sits below the floor."""
    values = sorted(cosine(cv, job) for job in jobs)
    above_floor = sum(1 for v in values if v > floor)
    above_ceiling = sum(1 for v in values if v >= ceiling)
    all_below = values[-1] < floor
    lines = [
        table("Synthetic CV (alex_example) against every distinct stored job", values),
        f"  jobs above the floor ({floor:.4f}):   {above_floor} of {len(values)}",
        f"  jobs at/above the ceiling ({ceiling:.4f}): {above_ceiling} of {len(values)}",
    ]
    return "\n".join(lines), all_below


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    print(f"Reading (read-only) from database {database_target(settings.database_url)}")
    vectors = await read_job_vectors(settings)
    if len(vectors) < 20:
        raise ScriptError(
            f"only {len(vectors)} jobs have a vector of {settings.embedding_model_id}: "
            "too few to calibrate (is the embedding sweep finished?)"
        )
    cosines = random_pair_cosines(vectors, args.pairs, args.seed)
    floor = round(percentile(cosines, FLOOR_PERCENTILE), 3)
    ceiling = round(percentile(cosines, CEILING_PERCENTILE), 3)
    print(f"model {settings.embedding_model_id}")
    print(
        f"{len(vectors)} jobs with distinct text, {len(cosines)} random pairs, seed {args.seed}\n"
    )
    print(table("Random job-job pairs (this is what floor and ceiling come from)", cosines))
    print(
        f"\nChosen: floor = P{FLOOR_PERCENTILE:g} = {floor}, "
        f"ceiling = P{CEILING_PERCENTILE:g} = {ceiling}"
    )
    if args.no_cv:
        return 0
    if settings.embedding_provider != "fastembed":
        print(
            "\nCV report skipped: set EMBEDDING_PROVIDER=fastembed (the real model) or pass --no-cv"
        )
        return 0
    cv_text = await asyncio.to_thread(Path(args.cv).read_text, encoding="utf-8")
    embedder = build_embedder(settings)
    if embedder is None:  # unreachable: the provider was checked above, kept for the type checker
        raise ScriptError("no embedder is configured")
    vector = await asyncio.to_thread(cv_vector, embedder, cv_text)
    text, all_below = report_cv(vector, vectors, floor, ceiling)
    print("\n" + text)
    if all_below:
        print(
            "\nSTOP: the whole distribution of the CV sits below the floor. This is reported, "
            "not tuned away: decide with the owner before any scoring work continues."
        )
        return STOP_EXIT_CODE
    print("\n(Statement only: nothing above was used to choose the floor or the ceiling.)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--pairs", type=int, default=5000, help="random job pairs (default 5000)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the pair sample (default 1)")
    parser.add_argument("--cv", default=str(DEFAULT_CV), help="the invented CV to report against")
    parser.add_argument("--no-cv", action="store_true", help="skip the CV report")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
