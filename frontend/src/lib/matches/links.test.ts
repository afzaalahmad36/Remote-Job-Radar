// @vitest-environment node
import { describe, expect, it } from "vitest";

import { safeMailLink, safeWebLink } from "./links";

describe("safeWebLink", () => {
  it("offers web links and says where they go", () => {
    expect(safeWebLink("https://jobs.example.com/apply?id=1")).toEqual({
      href: "https://jobs.example.com/apply?id=1",
      label: "Apply on jobs.example.com",
    });
    expect(safeWebLink("http://example.com")?.href).toBe("http://example.com/");
  });

  it("shows a look-alike name in its encoded form", () => {
    // The second letter is a Cyrillic "а", not the Latin one.
    expect(safeWebLink("https://pаypal.example")?.label).toContain("xn--");
  });

  it.each([
    null,
    "",
    "jobs.example.com",
    "/apply",
    "//evil.example",
    "javascript:alert(1)",
    "JavaScript:alert(1)",
    " javascript:alert(1)",
    "java\nscript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox",
    "file:///etc/passwd",
    "mailto:a@example.com",
    "https://bank.example@evil.example/",
    "https://user:pass@example.com/",
  ])("offers nothing for %j", (value) => {
    expect(safeWebLink(value)).toBeNull();
  });
});

describe("safeMailLink", () => {
  it("offers a single plain address", () => {
    expect(safeMailLink("jobs+web@mail.example.com")).toEqual({
      href: "mailto:jobs+web@mail.example.com",
      label: "Email jobs+web@mail.example.com",
    });
  });

  it.each([
    null,
    "",
    "not an address",
    "a@example",
    "a@example.com,b@example.com",
    "a@example.com;b@example.com",
    "a@example.com?subject=x",
    "a@example.com&bcc=b@example.com",
    "a%0Abcc:b@example.com",
    "a@example.com\nbcc:b@example.com",
    "a@example.com\n",
    "<a@example.com>",
    " a@example.com",
  ])("offers nothing for %j", (value) => {
    expect(safeMailLink(value)).toBeNull();
  });
});
