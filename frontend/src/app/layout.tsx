import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Remote Job Radar",
  description: "Remote jobs ranked against your CV.",
};

// System fonts only: no request to a font CDN.
export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white font-sans text-slate-900">{children}</body>
    </html>
  );
}
