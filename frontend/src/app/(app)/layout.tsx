import type { ReactNode } from "react";

import { SessionGate } from "@/components/session-gate";

// Signed-in pages are built for each request and never stored.
export const dynamic = "force-dynamic";

export default function SignedInLayout({ children }: Readonly<{ children: ReactNode }>) {
  return <SessionGate>{children}</SessionGate>;
}
