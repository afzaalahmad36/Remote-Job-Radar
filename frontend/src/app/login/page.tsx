import { AuthForm } from "@/components/auth-form";
import { messageFor } from "@/lib/messages";

export const dynamic = "force-dynamic";

// A fixed list: the address bar must not be able to put its own words on the page.
const NOTICES = new Set(["session_ended", "not_revoked"]);

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ notice?: string | string[] }>;
}) {
  const { notice } = await searchParams;
  const known = typeof notice === "string" && NOTICES.has(notice);
  return <AuthForm mode="login" notice={known ? messageFor(notice) : undefined} />;
}
