"""Invented job posts in five languages, used to fix the `looks_english` threshold.

Everything here is made up for this test. The threshold in `app/matching/scoring.py` was chosen
from these texts ALONE (see docs/MATCHING.md): no stored job was looked at. Each language has a
plain-prose post, and German, Spanish and Portuguese also have a bullet-style post that mixes in
English technology names (the hard case: a non-English post full of "Python, Kubernetes, Redis").
"""

ENGLISH_PROSE = (
    "We are looking for a backend engineer to join our small team. You will design and build "
    "APIs that are used by thousands of customers every day, and you will work closely with the "
    "product team to ship features that matter. We value clear communication, careful testing "
    "and a calm approach to incidents. Our stack is Python, FastAPI and PostgreSQL, and we deploy "
    "to the cloud with Docker. If you have shipped a service to production and you enjoy mentoring "
    "others, we would love to hear from you."
)

ENGLISH_BULLETS = (
    "Senior Platform Engineer\n"
    "Requirements:\n"
    "- Python, Go, Kubernetes, Terraform, AWS\n"
    "- PostgreSQL, Redis, Kafka\n"
    "- CI/CD with GitHub Actions\n"
    "- Experience with Prometheus and Grafana\n"
    "Nice to have:\n"
    "- Rust\n"
    "- gRPC\n"
    "Benefits: remote work, equity, 25 days of holiday and a learning budget for the whole team.\n"
    "Salary: 90,000 to 120,000 USD per year. Send your CV to the address below if this sounds "
    "like you."
)

ENGLISH_STUB = (
    "Backend Engineer at Acme Demo. Remote, full time. Python and Django. Apply with your CV."
)

GERMAN_PROSE = (
    "Wir suchen eine Backend-Entwicklerin oder einen Backend-Entwickler für unser kleines Team. "
    "Du entwickelst Schnittstellen, die jeden Tag von Tausenden Kunden genutzt werden, und "
    "arbeitest eng mit dem Produktteam zusammen, damit neue Funktionen schnell und zuverlässig "
    "live gehen. Uns sind klare Kommunikation, sorgfältiges Testen und ein ruhiger Umgang mit "
    "Störungen wichtig. Unser Stack besteht aus Python, FastAPI und PostgreSQL, und wir betreiben "
    "alles in der Cloud mit Docker. Wenn du schon einmal einen Dienst in Produktion gebracht hast "
    "und gerne Kolleginnen und Kollegen unterstützt, freuen wir uns auf deine Bewerbung."
)

GERMAN_BULLETS = (
    "Senior Platform Engineer (m/w/d)\n"
    "Deine Aufgaben:\n"
    "- Betrieb und Weiterentwicklung unserer Kubernetes-Cluster auf AWS\n"
    "- Automatisierung mit Terraform und GitHub Actions\n"
    "- Monitoring mit Prometheus und Grafana\n"
    "Dein Profil:\n"
    "- Mehrjährige Erfahrung mit Python oder Go\n"
    "- Gute Kenntnisse in PostgreSQL und Redis\n"
    "Wir bieten: Remote-Arbeit, 30 Urlaubstage, ein Weiterbildungsbudget und ein Gehalt zwischen "
    "70.000 und 90.000 Euro pro Jahr. Bewirb dich mit deinem Lebenslauf."
)

FRENCH_PROSE = (
    "Nous recherchons un ingénieur backend pour rejoindre notre petite équipe. Vous concevrez des "
    "interfaces utilisées chaque jour par des milliers de clients et vous travaillerez avec "
    "l'équipe produit pour livrer rapidement des fonctionnalités fiables. Nous apprécions une "
    "communication claire, des tests rigoureux et une gestion sereine des incidents. Notre socle "
    "technique est composé de Python, FastAPI et PostgreSQL, et nous déployons dans le cloud avec "
    "Docker. Si vous avez déjà mis un service en production et que vous aimez accompagner vos "
    "collègues, envoyez-nous votre candidature."
)

SPANISH_PROSE = (
    "Buscamos una persona desarrolladora backend para unirse a nuestro pequeño equipo. Diseñarás "
    "interfaces que utilizan miles de clientes cada día y trabajarás junto al equipo de producto "
    "para publicar funcionalidades fiables con rapidez. Valoramos la comunicación clara, las "
    "pruebas cuidadosas y la calma ante los incidentes. Nuestra pila tecnológica es Python, "
    "FastAPI y PostgreSQL, y desplegamos en la nube con Docker. Si ya has llevado un servicio a "
    "producción y disfrutas ayudando a tus compañeros, nos encantaría conocerte."
)

SPANISH_BULLETS = (
    "Ingeniero/a de Plataforma Senior\n"
    "Requisitos:\n"
    "- Experiencia con Python, Go y Kubernetes\n"
    "- Conocimientos de PostgreSQL, Redis y Kafka\n"
    "- CI/CD con GitHub Actions\n"
    "Ofrecemos: trabajo remoto, 25 días de vacaciones, presupuesto de formación y un salario de "
    "40.000 a 55.000 euros al año. Envía tu currículum."
)

PORTUGUESE_PROSE = (
    "Procuramos uma pessoa desenvolvedora backend para se juntar à nossa pequena equipa. Vais "
    "desenhar interfaces que são usadas todos os dias por milhares de clientes e trabalhar com a "
    "equipa de produto para entregar funcionalidades fiáveis com rapidez. Valorizamos a "
    "comunicação clara, os testes cuidadosos e a calma perante incidentes. A nossa stack é "
    "Python, FastAPI e PostgreSQL, e publicamos na nuvem com Docker. Se já colocaste um serviço "
    "em produção e gostas de ajudar os colegas, gostaríamos de conhecer-te."
)

PORTUGUESE_BULLETS = (
    "Engenheiro de Plataforma Sénior\n"
    "Requisitos:\n"
    "- Experiência com Python, Go e Kubernetes\n"
    "- Conhecimentos de PostgreSQL, Redis e Kafka\n"
    "- CI/CD com GitHub Actions\n"
    "Oferecemos: trabalho remoto, 25 dias de férias, orçamento para formação e um salário entre "
    "30.000 e 45.000 euros por ano. Envia o teu currículo."
)

ENGLISH_POSTS = {"prose": ENGLISH_PROSE, "bullets": ENGLISH_BULLETS}
NON_ENGLISH_POSTS = {
    "german prose": GERMAN_PROSE,
    "german bullets": GERMAN_BULLETS,
    "french prose": FRENCH_PROSE,
    "spanish prose": SPANISH_PROSE,
    "spanish bullets": SPANISH_BULLETS,
    "portuguese prose": PORTUGUESE_PROSE,
    "portuguese bullets": PORTUGUESE_BULLETS,
}
