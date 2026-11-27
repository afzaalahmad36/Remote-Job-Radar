import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import HomePage from "./page";

afterEach(cleanup);

describe("HomePage", () => {
  it("has one main landmark with the product name as its heading", () => {
    render(<HomePage />);

    expect(screen.getByRole("main")).toBeDefined();
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Remote Job Radar");
  });
});
