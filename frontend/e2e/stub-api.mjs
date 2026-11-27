// A small stand-in for the API, only for the tests in `session.spec.ts`. It copies the one rule
// that matters here: a refresh token works once, and using it again ends that whole session.
// It listens on this machine only and holds invented data.
import http from "node:http";
import { setTimeout as sleep } from "node:timers/promises";

const PORT = 8765;
const REFRESH_DELAY_MS = 500; // long enough for parallel requests to arrive during one refresh

let sessions; // session id -> { email, refresh, used: Set, ended }
let users;
let refreshCalls;
let down;
let counter = 0;

function reset() {
  sessions = new Map();
  users = new Set();
  refreshCalls = 0;
  down = false;
}
reset();

function issue(id) {
  counter += 1;
  const session = sessions.get(id);
  session.refresh = `refresh-${id}-${counter}`;
  return {
    access_token: `access-${id}-${counter}`,
    refresh_token: session.refresh,
    token_type: "bearer",
    expires_in: 1800,
  };
}

function newSession(email) {
  counter += 1;
  const id = `s${counter}`;
  sessions.set(id, { email, used: new Set(), ended: false });
  return issue(id);
}

const sessionOf = (token) => sessions.get(token?.split("-")[1]);

async function refresh(token) {
  refreshCalls += 1;
  await sleep(REFRESH_DELAY_MS);
  const session = sessionOf(token);
  if (!session || session.ended) return [401, {}];
  if (session.used.has(token) || token !== session.refresh) {
    session.ended = true; // reuse: the whole session ends
    return [401, {}];
  }
  session.used.add(token);
  return [200, issue(token.split("-")[1])];
}

async function answer(method, path, body, authorization) {
  if (path === "/__reset") return reset(), [204];
  if (path === "/__down") return (down = body.down), [204];
  if (path === "/__state") {
    return [200, { refreshCalls, ended: [...sessions.values()].map((s) => s.ended) }];
  }
  if (down) return [503, {}];
  if (path === "/api/v1/auth/register") return users.add(body.email), [201, { email: body.email }];
  if (path === "/api/v1/auth/login") {
    return users.has(body.email) ? [200, newSession(body.email)] : [401, {}];
  }
  if (path === "/api/v1/auth/refresh") return refresh(body.refresh_token);
  if (path === "/api/v1/auth/logout") {
    const session = sessionOf(body.refresh_token);
    if (session) session.ended = true;
    return [204];
  }
  if (path === "/api/v1/users/me" && method === "GET") {
    // As in the real API, an access token keeps working after sign-out.
    const session = sessionOf(authorization?.replace("Bearer ", ""));
    return session ? [200, { email: session.email }] : [401, {}];
  }
  return [404, {}];
}

http
  .createServer(async (request, response) => {
    let text = "";
    for await (const chunk of request) text += chunk;
    const [status, value] = await answer(
      request.method,
      request.url,
      text ? JSON.parse(text) : {},
      request.headers.authorization,
    );
    response.writeHead(status, { "Content-Type": "application/json" });
    response.end(value === undefined ? undefined : JSON.stringify(value));
  })
  .listen(PORT, "127.0.0.1");
