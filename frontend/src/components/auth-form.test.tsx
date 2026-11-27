import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AuthForm } from "./auth-form";

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  replace.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(cleanup);

function fillAndSubmit(button: string) {
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-enough" } });
  fireEvent.click(screen.getByRole("button", { name: button }));
}

describe("AuthForm", () => {
  it("sends the credentials as JSON and goes to the dashboard", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    render(<AuthForm mode="login" />);

    fillAndSubmit("Sign in");

    await waitFor(() => expect(replace).toHaveBeenCalledWith("/"));
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/auth/login");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "a@example.com",
      password: "long-enough",
    });
  });

  it("shows our own message for the answer's code, and allows another try", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ code: "email_taken" }), { status: 409 }),
    );
    render(<AuthForm mode="register" />);

    fillAndSubmit("Create account");

    expect((await screen.findByRole("alert")).textContent).toContain("already exists");
    expect(screen.getByRole("button", { name: "Create account" })).toHaveProperty(
      "disabled",
      false,
    );
    expect(replace).not.toHaveBeenCalled();
  });

  it("copes with no connection", async () => {
    fetchMock.mockRejectedValue(new TypeError("offline"));
    render(<AuthForm mode="login" />);

    fillAndSubmit("Sign in");

    expect((await screen.findByRole("alert")).textContent).toContain("not available");
  });

  it("would never put the password in the address bar without its script", () => {
    const { container } = render(<AuthForm mode="login" />);

    expect(container.querySelector("form")?.getAttribute("method")).toBe("post");
  });
});
