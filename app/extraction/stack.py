"""A dictionary of technologies and the ways job posts write them.

Matching works on whole words (tokens), never on substrings: "java" must not match "javascript",
and a bare "go" or "r" is too ambiguous to count (only "golang" does). Adding a technology is one
line here; nothing else changes.
"""

import re
from itertools import islice

MAX_STACK = 30  # most entries kept from any one source (merge.py caps a model's list with it)
MAX_LISTED = 12  # a post naming more distinct technologies is a recruiter's list, not one role
MAX_TOKENS = 4_000  # words examined per text
_TOKEN = re.compile(r"[a-z0-9.][a-z0-9+#._-]{0,30}", re.IGNORECASE)

# canonical name -> the lowercase spellings that mean it
_TECHNOLOGIES: dict[str, tuple[str, ...]] = {
    "Python": ("python", "python3"),
    "JavaScript": ("javascript", "js", "es6"),
    "TypeScript": ("typescript",),
    "Java": ("java",),
    "Kotlin": ("kotlin",),
    "Swift": ("swift", "swiftui"),
    "Go": ("golang",),
    "Rust": ("rust",),
    "C++": ("c++",),
    "C#": ("c#",),
    "PHP": ("php",),
    "Ruby": ("ruby",),
    "Scala": ("scala",),
    "Elixir": ("elixir",),
    "Erlang": ("erlang",),
    "Haskell": ("haskell",),
    "Clojure": ("clojure",),
    "Dart": ("dart",),
    "SQL": ("sql",),
    "React": ("react", "react.js", "reactjs"),
    "React Native": ("react native", "react-native"),
    "Next.js": ("next.js", "nextjs"),
    "Vue": ("vue", "vue.js", "vuejs"),
    "Nuxt": ("nuxt", "nuxt.js", "nuxtjs"),
    "Angular": ("angular", "angularjs"),
    "Svelte": ("svelte", "sveltekit"),
    "Node.js": ("node.js", "nodejs"),
    "Express": ("express.js", "expressjs"),
    "NestJS": ("nestjs", "nest.js"),
    "Deno": ("deno",),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi",),
    "Spring": ("spring boot", "springboot"),
    "Ruby on Rails": ("ruby on rails", "rails"),
    "Laravel": ("laravel",),
    "Symfony": ("symfony",),
    ".NET": (".net", "dotnet", "asp.net"),
    "Flutter": ("flutter",),
    "Tailwind CSS": ("tailwind", "tailwindcss"),
    "GraphQL": ("graphql",),
    "gRPC": ("grpc",),
    "HTML": ("html", "html5"),
    "CSS": ("css", "css3"),
    "Sass": ("sass", "scss"),
    "Redux": ("redux",),
    "Webpack": ("webpack",),
    "Vite": ("vite",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql", "mariadb"),
    "MongoDB": ("mongodb", "mongo"),
    "Redis": ("redis",),
    "SQLite": ("sqlite",),
    "Elasticsearch": ("elasticsearch", "opensearch"),
    "Kafka": ("kafka",),
    "RabbitMQ": ("rabbitmq",),
    "Celery": ("celery",),
    "Snowflake": ("snowflake",),
    "BigQuery": ("bigquery",),
    "dbt": ("dbt",),
    "Airflow": ("airflow",),
    "Spark": ("pyspark", "apache spark"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "AWS": ("aws",),
    "GCP": ("gcp", "google cloud"),
    "Azure": ("azure",),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Terraform": ("terraform",),
    "Ansible": ("ansible",),
    "Jenkins": ("jenkins",),
    "GitHub Actions": ("github actions",),
    "GitLab CI": ("gitlab ci", "gitlab-ci"),
    "Linux": ("linux",),
    "Nginx": ("nginx",),
    "Git": ("git",),
    "Prometheus": ("prometheus",),
    "Grafana": ("grafana",),
    "Figma": ("figma",),
    "Selenium": ("selenium",),
    "Cypress": ("cypress",),
    "Playwright": ("playwright",),
    "Jest": ("jest",),
    "Pytest": ("pytest",),
    "OpenAI": ("openai",),
    "LangChain": ("langchain",),
    "Unity": ("unity3d", "unity engine"),
    "Unreal Engine": ("unreal engine", "unreal"),
    "Solidity": ("solidity",),
    "Shopify": ("shopify",),
    "WordPress": ("wordpress",),
    "Salesforce": ("salesforce",),
    "Firebase": ("firebase",),
    "Supabase": ("supabase",),
}

_ALIASES: dict[str, str] = {
    alias: canonical for canonical, aliases in _TECHNOLOGIES.items() for alias in aliases
}
_CANONICAL_BY_FOLDED: dict[str, str] = {name.casefold(): name for name in _TECHNOLOGIES}


def canonical_name(name: str) -> str | None:
    """The dictionary's spelling of a technology the model or a post named, if we know it."""
    folded = name.strip().casefold()
    return _ALIASES.get(folded) or _CANONICAL_BY_FOLDED.get(folded)


def spellings(name: str) -> tuple[str, ...]:
    """The canonical name and every spelling listed for it, or () for an unknown technology."""
    canonical = canonical_name(name)
    return (canonical, *_TECHNOLOGIES[canonical]) if canonical else ()


# Spellings that are also ordinary English words. They count only when written with a capital
# letter ("Swift", "Rust"), so "swift cross-functional execution" is not a language.
AMBIGUOUS_SPELLINGS = frozenset("swift rust unreal ruby dart flask rails celery jest".split())

# A technology named only as somebody's customer or as the thing the product replaces is not
# part of this role's stack: "trusted by OpenAI, PayPal", "a Shopify alternative".
_LOOKBACK_CHARS = 120  # how far back a mention looks for its introducing words (one sentence)
_REFERENCE_BEFORE = re.compile(
    r"\b(?:trusted|used|loved|chosen) by\b"
    r"|\b(?:customers|clients|partners|investors)\s+(?:include|including|such as|like)\b"
    r"|\b(?:alternatives?|competitors?)\s+(?:to|of)\s+$"
    r"|\binstead of\s+$",
    re.IGNORECASE,
)
_REFERENCE_AFTER = re.compile(r"\W{0,3}(?:alternatives?|competitors?|clones?)\b", re.IGNORECASE)


def _is_reference(text: str, start: int, end: int) -> bool:
    """True if the mention sits in a customer list or an "X alternative" phrase."""
    window = text[max(0, start - _LOOKBACK_CHARS) : start]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("\n"))
    if _REFERENCE_BEFORE.search(window[cut + 1 :]):
        return True
    return _REFERENCE_AFTER.match(text, end, end + 20) is not None


def find_technologies(
    text: str, *, max_listed: int | None, skip_references: bool, max_tokens: int = MAX_TOKENS
) -> list[str]:
    """The scanner behind `find_stack` and the CV skills extractor.

    Technologies named in the text, in order of first appearance, without duplicates. The two
    callers differ only in these options: a job post drops customer-list mentions and gives up on
    a recruiter's list, a CV (written by the person themselves) does neither.

    Args:
        max_listed: more distinct technologies than this returns [] (None: no limit).
        skip_references: ignore a mention inside a customer list or an "X alternative" phrase.
        max_tokens: how many words of the text are examined.
    """
    tokens = [
        (match.group().lower().rstrip(".-_"), match.start())
        for match in islice(_TOKEN.finditer(text, 0, max_tokens * 12), max_tokens)
    ]
    found: dict[str, None] = {}
    for index in range(len(tokens)):
        for size in (3, 2, 1):  # prefer the longest spelling ("ruby on rails" over "ruby")
            chunk = tokens[index : index + size]
            phrase = " ".join(token for token, _ in chunk)
            if len(chunk) == size and phrase in _ALIASES:
                start = chunk[0][1]
                end = chunk[-1][1] + len(chunk[-1][0])
                capitalised = text[start].isupper()
                if (capitalised or phrase not in AMBIGUOUS_SPELLINGS) and not (
                    skip_references and _is_reference(text, start, end)
                ):
                    found.setdefault(_ALIASES[phrase], None)
                break
        if max_listed is not None and len(found) > max_listed:
            return []
    return list(found)


def find_stack(text: str) -> list[str]:
    """Technologies named in a job post, in order of first appearance, without duplicates.

    A mention inside a customer list or an "X alternative" phrase does not count. More than
    MAX_LISTED distinct technologies means the post is a category list ("we place developers in
    Python, Java, Ruby, ..."), not the stack of one role: an empty list is returned.
    """
    return find_technologies(text, max_listed=MAX_LISTED, skip_references=True)
