import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

// The privacy rules are errors, not conventions: a CV or a token must never end up in browser
// storage or in a log, and text from a job post must never be rendered as HTML.
const STORAGE = "Browser storage is not used: CV text and tokens must never be kept there.";
const STORES = ["localStorage", "sessionStorage", "indexedDB"];

const privacyRules = {
  "no-console": "error",
  "no-restricted-globals": ["error", ...STORES.map((name) => ({ name, message: STORAGE }))],
  "no-restricted-properties": [
    "error",
    ...["window", "globalThis"].flatMap((object) =>
      STORES.map((property) => ({ object, property, message: STORAGE })),
    ),
  ],
  "react/no-danger": "error",
};

export default defineConfig([
  ...nextVitals,
  ...nextTypescript,
  { rules: privacyRules },
  {
    // Components run in the browser: server code (tokens, cookies) must not be pulled into them.
    files: ["src/components/**"],
    rules: {
      "no-restricted-imports": ["error", { patterns: ["**/lib/server/*"] }],
    },
  },
  globalIgnores([
    ".next/**",
    "next-env.d.ts",
    "src/lib/api/schema.d.ts",
    "test-results/**",
    "playwright-report/**",
  ]),
]);
