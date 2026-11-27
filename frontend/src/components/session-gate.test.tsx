import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SessionGate } from "./session-gate";

const router = { replace: vi.fn() }; // one object, as in Next.js: the effect depends on it
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const fetchMock = vi.fn<typeof fetch>();
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const signedIn = () => json(200, { email: "a@example.com" });

beforeEach(() => {
  router.replace.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(cleanup);

const open = () =>
  render(
    <SessionGate>
      <p>private page</p>
    </SessionGate>,
  );

describe("SessionGate", () => {
  it("shows the page only once the user is known", async () => {
    fetchMock.mockResolvedValue(signedIn());

    open();

    expect(screen.queryByText("private page")).toBeNull();
    expect(await screen.findByText("private page")).toBeDefined();
    expect(screen.getByText("a@example.com")).toBeDefined();
  });

  it.each([
    ["not_signed_in", "/login"],
    ["session_ended", "/login?notice=session_ended"],
  ])("sends a 401 %s to %s", async (code, target) => {
    fetchMock.mockResolvedValue(json(401, { code }));

    open();

    await waitFor(() => expect(router.replace).toHaveBeenCalledWith(target));
    expect(screen.queryByText("private page")).toBeNull();
  });

  it("offers another try when the service is down, without signing out", async () => {
    fetchMock.mockResolvedValueOnce(json(502, { code: "api_unavailable" }));
    open();
    await screen.findByRole("alert");
    fetchMock.mockResolvedValue(signedIn());

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByText("private page")).toBeDefined();
    expect(router.replace).not.toHaveBeenCalled();
  });

  it.each([
    [() => new Response(null, { status: 204 }), "/login"],
    [() => json(502, { code: "not_revoked" }), "/login?notice=not_revoked"],
  ])("signs out", async (answer, target) => {
    fetchMock.mockResolvedValueOnce(signedIn()).mockResolvedValueOnce(answer());
    open();

    fireEvent.click(await screen.findByRole("button", { name: "Sign out" }));

    await waitFor(() => expect(router.replace).toHaveBeenCalledWith(target));
    expect(fetchMock.mock.calls[1]).toEqual(["/api/auth/logout", { method: "POST" }]);
  });

  it("does not claim to be signed out when the request failed", async () => {
    fetchMock.mockResolvedValueOnce(signedIn()).mockRejectedValueOnce(new TypeError("offline"));
    open();

    fireEvent.click(await screen.findByRole("button", { name: "Sign out" }));

    expect((await screen.findByRole("alert")).textContent).toContain("did not work");
    expect(router.replace).not.toHaveBeenCalled();
  });
});
