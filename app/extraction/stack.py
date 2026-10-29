"""A dictionary of technologies and the ways job posts write them.

Matching works on whole words (tokens), never on substrings: "java" must not match "javascript",
and a bare "go" or "r" is too ambiguous to count (only "golang" does). Adding a technology is one
line here; nothing else changes.
"""

import re

MAX_STACK = 30
MAX_TOKENS = 4_000  # words examined per text
_TOKEN = re.compile(r"[a-z0-9.][a-z0-9+#._-]{0,30}")

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


def find_stack(text: str) -> list[str]:
    """Technologies named in the text, in order of first appearance, without duplicates."""
    tokens = [
        token.rstrip(".-_")
        for token in _TOKEN.findall(text.lower()[: MAX_TOKENS * 12])[:MAX_TOKENS]
    ]
    found: dict[str, None] = {}
    for index in range(len(tokens)):
        for size in (3, 2, 1):  # prefer the longest spelling ("ruby on rails" over "ruby")
            phrase = " ".join(tokens[index : index + size])
            if len(tokens[index : index + size]) == size and phrase in _ALIASES:
                found.setdefault(_ALIASES[phrase], None)
                break
        if len(found) >= MAX_STACK:
            break
    return list(found)
