import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { type Match, type MatchPage, salary, warnings } from "@/lib/matches/format";

import { MatchFilters } from "./match-filters";
import { MatchList } from "./match-list";

const router = { replace: vi.fn(), push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const fetchMock = vi.fn<typeof fetch>();
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const NO_FILTERS = { min: null, all: false, page: 1 };

function match(changes: Partial<Match> = {}, job: Partial<Match["job"]> = {}): Match {
  return {
    job: {
      id: "3f2b8c1e-9a4d-4e6f-8b2a-1c5d7e9f0a3b",
      source_id: "3f2b8c1e-9a4d-4e6f-8b2a-1c5d7e9f0a3c",
      title: "Backend Engineer",
      company: "Example Co",
      stack: ["Python", "Docker"],
      seniority: "senior",
      remote_policy: "remote",
      salary_min: 90000,
      salary_max: 120000,
      salary_currency: "USD",
      salary_period: "year",
      posted_at: "2026-11-05T23:30:00Z",
      apply_email: "jobs@example.com",
      apply_url: "https://example.com/apply",
      ...job,
    },
    score: 75,
    filters_failed: false,
    missing_skills: ["Docker"],
    scorer_version: "score_v1",
    updated_at: "2026-11-06T00:00:00Z",
    note_codes: ["no_embedding"],
    ...changes,
  };
}

function page(items: Match[], changes: Partial<MatchPage> = {}): MatchPage {
  const scoring = {
    state: "up_to_date" as const,
    requested_at: null,
    finished_at: null,
    failure_count: 0,
    max_attempts: 3,
    error: null,
  };
  return { items, total: items.length, limit: 20, offset: 0, scoring, ...changes };
}

const answer = (body: MatchPage) => fetchMock.mockResolvedValue(json(200, body));

beforeEach(() => {
  router.replace.mockReset();
  router.push.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(cleanup);

describe("MatchList", () => {
  it("shows a match with its score, facts, warnings, skills and links", async () => {
    answer(page([match()]));

    render(<MatchList filters={NO_FILTERS} />);

    const card = within(await screen.findByRole("article"));
    expect(card.getByRole("heading").textContent).toBe("Backend Engineer");
    expect(card.getByText("75")).toBeDefined();
    expect(
      card.getByText(
        "Example Co · Remote · Senior · USD 90,000–120,000 per year · Posted 5 Nov 2026",
      ),
    ).toBeDefined();
    expect(card.getByRole("list", { name: "Warnings" }).textContent).toBe("Skills only");
    const skills = card.getByRole("list", { name: "Skills the job lists" });
    expect([...skills.children].map((item) => item.textContent)).toEqual([
      "Python",
      "Docker (missing)",
    ]);
    const apply = card.getByRole("link", { name: "Apply on example.com" });
    expect(apply.getAttribute("rel")).toBe("noopener noreferrer");
    expect(apply.getAttribute("target")).toBe("_blank");
    expect(card.getByRole("link", { name: "Email jobs@example.com" })).toBeDefined();
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/v1/matches?limit=20&offset=0");
  });

  it("shows hostile post text as text, and offers no hostile link", async () => {
    const hostile = '<img src=x onerror="alert(1)">';
    answer(
      page([
        match(
          {},
          {
            title: hostile,
            apply_url: "javascript:alert(1)",
            apply_email: "a@example.com?bcc=b@example.com",
          },
        ),
      ]),
    );

    const { container } = render(<MatchList filters={NO_FILTERS} />);

    expect((await screen.findByRole("heading")).textContent).toBe(hostile);
    expect(container.querySelector("img")).toBeNull();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it.each([
    [page([], { scoring: { ...page([]).scoring, state: "no_cv" } }), NO_FILTERS, "Upload a CV"],
    [page([]), NO_FILTERS, "Jobs that fail your preferences are hidden"],
    [page([]), { min: 90, all: true, page: 1 }, "Clear the filters"],
  ])("explains an empty list", async (body, filters, words) => {
    answer(body);

    const { container } = render(<MatchList filters={filters} />);

    await vi.waitFor(() => expect(container.textContent).toContain(words));
  });

  it("links to the other pages, keeping the filters", async () => {
    answer(page([match()], { total: 45 }));

    render(<MatchList filters={{ min: 50, all: false, page: 2 }} />);

    expect(await screen.findByText("Page 2 of 3")).toBeDefined();
    expect(screen.getByRole("link", { name: "Previous" }).getAttribute("href")).toBe("/?min=50");
    expect(screen.getByRole("link", { name: "Next" }).getAttribute("href")).toBe(
      "/?min=50&page=3",
    );
  });

  it("sends a lost session to sign in", async () => {
    fetchMock.mockResolvedValue(json(401, { code: "session_ended" }));

    render(<MatchList filters={NO_FILTERS} />);

    await vi.waitFor(() =>
      expect(router.replace).toHaveBeenCalledWith("/login?notice=session_ended"),
    );
  });

  it("offers another try after a failure", async () => {
    fetchMock.mockResolvedValueOnce(json(502, { code: "api_unavailable" }));
    render(<MatchList filters={NO_FILTERS} />);
    await screen.findByRole("alert");
    answer(page([match()]));

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByRole("article")).toBeDefined();
  });

  it("cancels its request when the page is left", () => {
    fetchMock.mockReturnValue(new Promise(() => undefined));
    const { unmount } = render(<MatchList filters={NO_FILTERS} />);

    unmount();

    expect(fetchMock.mock.calls[0]![1]?.signal?.aborted).toBe(true);
  });
});

describe("MatchFilters", () => {
  it("puts the chosen filters into the address, from page 1", () => {
    render(<MatchFilters filters={{ min: null, all: false, page: 3 }} />);

    fireEvent.change(screen.getByLabelText("Minimum score"), { target: { value: "60" } });
    fireEvent.click(screen.getByLabelText("Show jobs that fail my filters"));
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));

    expect(router.push).toHaveBeenCalledWith("/?min=60&all=1");
  });
});

describe("wording", () => {
  it("warns for every note code, and says when the explanation cannot be read", () => {
    const all = match({
      note_codes: [
        "filters_failed",
        "no_job_stack",
        "no_cv_skills",
        "no_embedding",
        "not_english",
        "skills_truncated",
      ],
    });

    expect(warnings(all)).toHaveLength(6);
    expect(warnings(all).every(Boolean)).toBe(true);
    expect(warnings(match({ note_codes: [] }))).toEqual([]);
    expect(warnings(match({ note_codes: null }))).toEqual(["Explanation outdated"]);
    expect(warnings(match({ note_codes: null, filters_failed: true }))).toEqual([
      "Fails a filter",
      "Explanation outdated",
    ]);
  });

  it.each([
    [{ salary_min: null, salary_max: null }, null],
    [{ salary_min: null }, "USD up to 120,000 per year"],
    [{ salary_max: null }, "USD from 90,000 per year"],
    [{ salary_max: 90000 }, "USD 90,000 per year"],
    [{ salary_currency: null, salary_period: null }, "90,000–120,000"],
  ])("words a salary %j", (changes, words) => {
    expect(salary(match({}, changes).job)).toBe(words);
  });
});
