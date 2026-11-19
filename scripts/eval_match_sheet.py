"""Build the blind labelling sheets and the answer key of the matching evaluation.

Run by hand, never by CI or app code. The protocol is docs/MATCHING.md, section 10, and was fixed
before any sheet existed. Read it first.

What it does, in order:

1. reads the frozen pool from the development database (read-only, only a database named
   `jobradar`): extracted jobs with a vector of the current model, one per distinct post text;
2. refuses to go on unless the pool is exactly the one recorded in the protocol (count and hash);
3. makes each persona's CV vector like an upload does (clean, mask, chunk, embed, mean), so it needs
   the real model: EMBEDDING_PROVIDER=fastembed, i.e. the Docker image with the network off;
4. ranks the pool five ways, picks the sheet items, shuffles them with a fixed seed;
5. writes one sheet per persona (`<out>/sheets/<persona>.json`: public job fields only, never a
   rank, a score or a method) and ONE answer key (`<out>/key/key.json`) in a different directory.

`--dry-run` does steps 1 to 4 and prints counts only: no job id, title, skill or CV text, and it
writes nothing. A real run refuses to overwrite an existing sheet or key (delete them yourself: a
rebuilt sheet would no longer match labels already made).

No provider request is made, no LLM client is built, nothing is written to the database. The
personas come from a TOML file (default: the invented ones under tests/fixtures/cv); the owner's own
CV may be added with `--real-personas` pointing at a file under `.local/`.

    uv run python -m scripts.eval_match_sheet --dry-run
    uv run python -m scripts.eval_match_sheet [--seed 1] [--out .local/match_eval]
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.cv.skills import find_skills
from app.cv.text import clean_cv_text, mask_cv_text
from app.matching.embeddings import Embedder
from app.matching.factory import build_embedder
from app.matching.scoring import ScoringConfig
from app.models.job import ExtractionStatus, Job
from app.services.match_service import scoring_config
from scripts.calibrate_similarity import cv_vector
from scripts.db_guard import ScriptError, database_target, ensure_jobs_table, read_only_engine
from scripts.match_eval import (
    DEFAULT_SEED,
    PROTOCOL_POOL_SIZE,
    SHEET_TEXT_CHARS,
    Persona,
    PersonaPlan,
    PersonaSpec,
    PoolJob,
    check_pool,
    describe,
    key_document,
    load_personas,
    plan_all,
    sheet_document,
)

DEFAULT_PERSONAS = Path("tests/fixtures/cv/personas.toml")
LOCAL_DIR = Path(".local")  # the git-ignored folder: the only place a real CV may be named
DEFAULT_OUT = LOCAL_DIR / "match_eval"


async def read_pool(settings: Settings) -> list[PoolJob]:
    """The pool: extracted jobs with a vector of the current model, one per distinct text."""
    engine = read_only_engine(settings.database_url)
    try:
        async with engine.connect() as connection:
            await ensure_jobs_table(connection)
            rows = await connection.execute(
                select(
                    Job.id,
                    Job.title,
                    Job.company,
                    Job.seniority,
                    Job.remote_policy,
                    Job.stack,
                    Job.salary_min,
                    Job.salary_max,
                    Job.salary_currency,
                    Job.salary_period,
                    Job.posted_at,
                    Job.created_at,
                    func.left(Job.raw_text, SHEET_TEXT_CHARS),
                    Job.raw_text_hash,
                    Job.extraction_rules_version,
                    Job.embedding,
                )
                .where(
                    Job.extraction_status == ExtractionStatus.EXTRACTED,
                    Job.embedding.is_not(None),
                    Job.embedding_model == settings.embedding_model_id,
                    Job.raw_text_hash.is_not(None),
                )
                .distinct(Job.raw_text_hash)
                .order_by(Job.raw_text_hash, Job.id)
            )
            return [
                PoolJob(
                    id=str(r[0]),
                    title=r[1] or "",
                    company=r[2],
                    seniority=r[3],
                    remote_policy=r[4],
                    stack=tuple(r[5] or ()),
                    salary_min=r[6],
                    salary_max=r[7],
                    salary_currency=r[8],
                    salary_period=r[9],
                    posted_at=r[10],
                    created_at=r[11],
                    text=r[12] or "",
                    raw_text_hash=r[13],
                    rules_version=r[14],
                    vector=[float(x) for x in r[15]],
                )
                for r in rows
            ]
    finally:
        await engine.dispose()


def prepare_personas(specs: Sequence[PersonaSpec], embedder: Embedder) -> list[Persona]:
    """Skills and CV vector exactly as an upload makes them. The CV text goes no further."""
    personas = []
    for spec in specs:
        masked = mask_cv_text(clean_cv_text(spec.cv_text))
        personas.append(
            Persona(
                name=spec.name,
                kind=spec.kind,
                skills=tuple(find_skills(masked)),
                vector=cv_vector(embedder, spec.cv_text),
                preferences=spec.preferences,
            )
        )
    return personas


def build(
    pool: Sequence[PoolJob],
    specs: Sequence[PersonaSpec],
    embedder: Embedder,
    config: ScoringConfig,
    seed: int,
) -> list[PersonaPlan]:
    """Check the pool, prepare the personas and plan every sheet. Writes nothing."""
    check_pool(pool)
    return plan_all(prepare_personas(specs, embedder), pool, config, seed)


def write_outputs(
    out: Path,
    plans: Sequence[PersonaPlan],
    pool: Sequence[PoolJob],
    config: ScoringConfig,
    seed: int,
) -> list[Path]:
    """Sheets and key in different directories; never overwrites a file."""
    targets = {out / "sheets" / f"{p.name}.json": sheet_document(p, pool) for p in plans}
    targets[out / "key" / "key.json"] = key_document(plans, pool, config, seed)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise ScriptError(
            "refusing to overwrite: " + ", ".join(p.name for p in existing) + " already exist(s); "
            "delete them yourself (labels made on an old sheet would not match a new one)"
        )
    for path, document in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
    return list(targets)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS)
    parser.add_argument(
        "--real-personas", type=Path, help="optional persona file with the owner's own CV (.local/)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print counts only, write nothing")
    parser.add_argument(
        "--allow-provider", action="store_true", help="refused: no provider is used"
    )
    return parser.parse_args(argv)


def check_out_dir(out: Path, local_root: Path = LOCAL_DIR) -> None:
    """Sheets, key and labels name jobs and show post text: they stay in the git-ignored folder."""
    if not out.resolve().is_relative_to(local_root.resolve()):
        raise ScriptError("--out must be inside .local/ (it is git-ignored)")


async def run(args: argparse.Namespace) -> int:
    if args.allow_provider:
        raise ScriptError("this script never asks a provider; --allow-provider is refused")
    check_out_dir(args.out)
    settings = get_settings()
    if settings.embedding_provider != "fastembed":
        raise ScriptError("needs the real model: set EMBEDDING_PROVIDER=fastembed (Docker image)")
    print(f"database: {database_target(settings.database_url)} (read-only)")
    specs = load_personas(args.personas)
    if args.real_personas is not None:
        specs += load_personas(args.real_personas, local_root=LOCAL_DIR)
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ScriptError("persona names must be unique across both files")
    pool = await read_pool(settings)
    print(f"pool read: {len(pool)} jobs (the protocol froze {PROTOCOL_POOL_SIZE})")
    embedder = build_embedder(settings)
    if embedder is None:
        raise ScriptError("no embedder is configured")
    config = scoring_config(settings)
    plans = build(pool, specs, embedder, config, args.seed)
    print("pool hash: matches the protocol")
    for line in describe(plans, pool):
        print(line)
    if args.dry_run:
        print("dry run: nothing was written")
        return 0
    written = write_outputs(args.out, plans, pool, config, args.seed)
    print(f"wrote {len(written)} files under {args.out} (sheets/ and key/ are separate)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(run(parse_args(argv)))
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
