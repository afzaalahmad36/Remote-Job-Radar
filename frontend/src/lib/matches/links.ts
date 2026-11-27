// An apply link or email comes from a job post, which anyone can write. These two functions decide
// what may be clickable; everything else is not offered.

export interface SafeLink {
  href: string;
  label: string; // says where the link goes, so the reader can judge it
}

/** An ordinary web link, or null. `javascript:`, `data:` and the like never pass. */
export function safeWebLink(value: string | null): SafeLink | null {
  if (!value || !URL.canParse(value)) return null;
  const url = new URL(value);
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  // "https://bank.example@evil.example" goes to evil.example: refuse the disguise.
  if (url.username || url.password) return null;
  return { href: url.href, label: `Apply on ${url.hostname}` };
}

// One address and nothing else: no spaces, commas, "?", "&" or "%", any of which could add more
// recipients, a subject or a body to the email that opens.
const EMAIL = /^[A-Za-z0-9._+-]{1,64}@[A-Za-z0-9.-]{1,250}\.[A-Za-z]{2,24}$/;

export function safeMailLink(value: string | null): SafeLink | null {
  if (!value || !EMAIL.test(value)) return null;
  return { href: `mailto:${value}`, label: `Email ${value}` };
}
