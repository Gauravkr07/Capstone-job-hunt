/*
 * JobHunt front door: sign in / create account, then a small home screen.
 * React 18 via UMD + htm for JSX-like templates -- no build step, served by
 * FastAPI from /static. Routes are hash-based (#/login, #/register, #/home)
 * so they never collide with the POST /login and POST /register API routes.
 */
(function () {
  const { useState, useEffect, useCallback } = React;
  const html = htm.bind(React.createElement);

  // ---------------------------------------------------------------- session

  const SESSION_KEY = "jobhunt_session";

  function loadSession() {
    try {
      const session = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
      if (session && session.jwt && session.expiresAt * 1000 > Date.now()) return session;
    } catch (_) {
      /* storage blocked or corrupt -- treat as signed out */
    }
    return null;
  }

  function saveSession(session) {
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(session));
      // The chatbot page reads this key to tag its conversations.
      localStorage.setItem("jobhunt_username", session.username);
    } catch (_) {
      /* session still works for this page view */
    }
  }

  function clearSession() {
    try {
      localStorage.removeItem(SESSION_KEY);
    } catch (_) {}
  }

  // -------------------------------------------------------------------- api

  class ApiError extends Error {
    constructor(status, message, fieldErrors) {
      super(message);
      this.status = status;
      this.fieldErrors = fieldErrors || {};
    }
  }

  // FastAPI returns {detail: "msg"} for HTTPException and
  // {detail: [{loc: [..., field], msg}]} for validation (422) errors.
  function toApiError(status, body) {
    const detail = body && (body.detail ?? body.message);
    if (Array.isArray(detail)) {
      const fieldErrors = {};
      detail.forEach((item) => {
        const field = item.loc ? item.loc[item.loc.length - 1] : "form";
        fieldErrors[field] = (item.msg || "Invalid value").replace(/^Value error, /, "");
      });
      return new ApiError(status, "Please fix the highlighted fields.", fieldErrors);
    }
    if (typeof detail === "string") return new ApiError(status, detail);
    return new ApiError(status, status >= 500 ? "The server had a problem. Try again shortly." : "Request failed.");
  }

  async function postJson(path, payload) {
    let response;
    try {
      response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (_) {
      throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
    }
    let body = null;
    try {
      body = await response.json();
    } catch (_) {}
    if (!response.ok) throw toApiError(response.status, body);
    return body;
  }

  async function login(username, password) {
    const body = await postJson("/login", { username, password });
    const session = { jwt: body.jwt, username: username.trim().toLowerCase(), expiresAt: body.jwt_expires_at };
    saveSession(session);
    return session;
  }

  // ----------------------------------------------------------------- router

  function useHashRoute() {
    const read = () => window.location.hash.replace(/^#\/?/, "").split("?")[0];
    const [route, setRoute] = useState(read);
    useEffect(() => {
      const onChange = () => setRoute(read());
      window.addEventListener("hashchange", onChange);
      return () => window.removeEventListener("hashchange", onChange);
    }, []);
    return route;
  }

  function go(route) {
    window.location.hash = "#/" + route;
  }

  // ------------------------------------------------------------- primitives

  function Field({ id, label, hint, error, type = "text", value, onChange, autoComplete, required, ...rest }) {
    const [revealed, setRevealed] = useState(false);
    const isPassword = type === "password";
    return html`
      <div className="field">
        <label htmlFor=${id}>${label}${hint ? html` <span className="hint">${hint}</span>` : null}</label>
        <div className="input-wrap">
          <input
            id=${id}
            name=${id}
            type=${isPassword && revealed ? "text" : type}
            value=${value}
            onChange=${(e) => onChange(e.target.value)}
            autoComplete=${autoComplete}
            required=${required}
            aria-invalid=${error ? "true" : "false"}
            aria-describedby=${error ? id + "-error" : undefined}
            style=${isPassword ? { paddingRight: "64px" } : undefined}
            ...${rest}
          />
          ${isPassword &&
          html`<button
            type="button"
            className="reveal"
            onClick=${() => setRevealed(!revealed)}
            aria-label=${revealed ? "Hide password" : "Show password"}
          >
            ${revealed ? "Hide" : "Show"}
          </button>`}
        </div>
        ${error && html`<span className="error" id=${id + "-error"}>${error}</span>`}
      </div>
    `;
  }

  function Switch({ id, label, help, checked, onChange }) {
    return html`
      <div className="switch-row">
        <div>
          <div className="label" id=${id + "-label"}>${label}</div>
          ${help && html`<div className="help">${help}</div>`}
        </div>
        <button
          type="button"
          role="switch"
          className="switch"
          aria-checked=${checked ? "true" : "false"}
          aria-labelledby=${id + "-label"}
          onClick=${() => onChange(!checked)}
        ></button>
      </div>
    `;
  }

  function SubmitButton({ busy, children }) {
    return html`
      <button type="submit" className="btn" disabled=${busy} aria-busy=${busy ? "true" : "false"}>
        ${busy && html`<span className="spinner" aria-hidden="true"></span>`}
        ${children}
      </button>
    `;
  }

  function Alert({ kind = "error", children }) {
    if (!children) return null;
    return html`<div className=${"alert " + kind} role=${kind === "error" ? "alert" : "status"}>${children}</div>`;
  }

  // -------------------------------------------------------------- auth pages

  function LoginForm({ onSignedIn, notice }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");

    async function submit(e) {
      e.preventDefault();
      if (!username.trim() || !password) {
        setError("Enter your username and password.");
        return;
      }
      setBusy(true);
      setError("");
      try {
        onSignedIn(await login(username, password));
      } catch (err) {
        setError(err.status === 401 ? "That username and password don't match." : err.message);
      } finally {
        setBusy(false);
      }
    }

    return html`
      <form onSubmit=${submit} noValidate>
        <${Alert} kind="success">${notice}<//>
        <${Alert}>${error}<//>
        <${Field} id="username" label="Username" value=${username} onChange=${setUsername} autoComplete="username" required autoFocus />
        <${Field} id="password" label="Password" type="password" value=${password} onChange=${setPassword} autoComplete="current-password" required />
        <${SubmitButton} busy=${busy}>Sign in<//>
        <p className="switch-link">New to JobHunt? <a href="#/register">Create an account</a></p>
      </form>
    `;
  }

  const EMPTY_REGISTRATION = {
    username: "",
    full_name: "",
    email: "",
    password: "",
    confirm: "",
    age: "",
    graduated: false,
    linkedin_url: "",
    leetcode_url: "",
    hackerrank_url: "",
  };

  function validateRegistration(form) {
    const errors = {};
    if (!/^[a-zA-Z0-9_.-]{3,150}$/.test(form.username.trim())) {
      errors.username = "Use 3 or more letters, numbers, dots, dashes or underscores.";
    }
    if (form.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) errors.email = "Enter a valid email address.";
    if (form.password.length < 8) errors.password = "Use at least 8 characters.";
    if (form.confirm !== form.password) errors.confirm = "Passwords don't match.";
    const age = Number(form.age);
    if (!Number.isInteger(age) || age < 14 || age > 100) errors.age = "Enter your age in years.";
    ["linkedin_url", "leetcode_url", "hackerrank_url"].forEach((key) => {
      if (form[key] && !/^https?:\/\/\S+$/.test(form[key])) errors[key] = "Enter a full link starting with https://";
    });
    return errors;
  }

  function RegisterForm({ onSignedIn }) {
    const [form, setForm] = useState(EMPTY_REGISTRATION);
    const [showLinks, setShowLinks] = useState(false);
    const [errors, setErrors] = useState({});
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);

    const set = useCallback((key) => (value) => setForm((f) => ({ ...f, [key]: value })), []);

    async function submit(e) {
      e.preventDefault();
      const found = validateRegistration(form);
      setErrors(found);
      if (Object.keys(found).length) {
        setError("Please fix the highlighted fields.");
        return;
      }
      setBusy(true);
      setError("");
      const optional = (v) => (v && v.trim() ? v.trim() : null);
      try {
        await postJson("/register", {
          username: form.username.trim(),
          password: form.password,
          full_name: optional(form.full_name),
          email: optional(form.email),
          age: Number(form.age),
          graduated: form.graduated,
          linkedin_url: showLinks ? optional(form.linkedin_url) : null,
          leetcode_url: showLinks ? optional(form.leetcode_url) : null,
          hackerrank_url: showLinks ? optional(form.hackerrank_url) : null,
        });
        // Account created -- sign straight in so the user lands on home.
        onSignedIn(await login(form.username, form.password));
      } catch (err) {
        if (err.status === 400 && /exists/i.test(err.message)) {
          setErrors({ username: "That username is taken." });
          setError("That username is taken. Pick another or sign in.");
        } else {
          setErrors(err.fieldErrors);
          setError(err.message);
        }
      } finally {
        setBusy(false);
      }
    }

    return html`
      <form onSubmit=${submit} noValidate>
        <${Alert}>${error}<//>
        <div className="row">
          <${Field} id="full_name" label="Full name" hint="(optional)" value=${form.full_name} onChange=${set("full_name")} autoComplete="name" error=${errors.full_name} />
          <${Field} id="username" label="Username" value=${form.username} onChange=${set("username")} autoComplete="username" error=${errors.username} required />
        </div>
        <${Field} id="email" label="Email" hint="(optional)" type="email" value=${form.email} onChange=${set("email")} autoComplete="email" error=${errors.email} />
        <div className="row">
          <${Field} id="password" label="Password" type="password" value=${form.password} onChange=${set("password")} autoComplete="new-password" error=${errors.password} required />
          <${Field} id="confirm" label="Confirm password" type="password" value=${form.confirm} onChange=${set("confirm")} autoComplete="new-password" error=${errors.confirm} required />
        </div>
        <div className="row">
          <${Field} id="age" label="Age" type="number" inputMode="numeric" min="14" max="100" value=${form.age} onChange=${set("age")} error=${errors.age} required />
          <div></div>
        </div>
        <${Switch} id="graduated" label="I have graduated" help="Tell recruiters whether you've finished your degree." checked=${form.graduated} onChange=${set("graduated")} />
        <${Switch} id="links" label="Add profile links" help="LinkedIn, LeetCode and HackerRank. You can skip this." checked=${showLinks} onChange=${setShowLinks} />
        ${showLinks &&
        html`<div className="optional">
          <${Field} id="linkedin_url" label="LinkedIn" type="url" placeholder="https://linkedin.com/in/…" value=${form.linkedin_url} onChange=${set("linkedin_url")} error=${errors.linkedin_url} />
          <${Field} id="leetcode_url" label="LeetCode" type="url" placeholder="https://leetcode.com/…" value=${form.leetcode_url} onChange=${set("leetcode_url")} error=${errors.leetcode_url} />
          <${Field} id="hackerrank_url" label="HackerRank" type="url" placeholder="https://hackerrank.com/…" value=${form.hackerrank_url} onChange=${set("hackerrank_url")} error=${errors.hackerrank_url} />
        </div>`}
        <${SubmitButton} busy=${busy}>Create account<//>
        <p className="switch-link">Already have an account? <a href="#/login">Sign in</a></p>
      </form>
    `;
  }

  function Check() {
    return html`<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8fa9f5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle><path d="M8 12.5l2.5 2.5L16 9.5"></path></svg>`;
  }

  function AuthPage({ mode, onSignedIn, notice }) {
    const isLogin = mode === "login";
    useEffect(() => {
      document.title = isLogin ? "JobHunt — Sign in" : "JobHunt — Create account";
    }, [isLogin]);

    return html`
      <div className="auth">
        <aside className="brand">
          <div className="logo">Job<span>Hunt</span></div>
          <div>
            <h1>Find the jobs your CV actually fits.</h1>
            <p>Upload your CV once. We match it against every posted job and show you only the ones worth applying to.</p>
          </div>
          <ul className="features">
            <li><${Check} /><div><strong>One CV, always current</strong><span>Paste it, link it or upload a file. A new upload replaces the old one everywhere.</span></div></li>
            <li><${Check} /><div><strong>Matches at 45% and above</strong><span>Scored on your skills and overall profile, best fit first.</span></div></li>
            <li><${Check} /><div><strong>Know your gaps</strong><span>Paste any job description to see what you have and what's missing.</span></div></li>
          </ul>
          <div className="brand-foot">Capstone Job Hunt</div>
        </aside>

        <main className="panel">
          <div className="card">
            <div className="logo mobile-logo">Job<span>Hunt</span></div>
            <div className="card-head">
              <h2>${isLogin ? "Welcome back" : "Create your account"}</h2>
              <p>${isLogin ? "Sign in to see your job matches." : "It takes a minute. You'll add your CV next."}</p>
            </div>
            <nav className="tabs" aria-label="Account">
              <a href="#/login" aria-current=${isLogin ? "page" : undefined}>Sign in</a>
              <a href="#/register" aria-current=${isLogin ? undefined : "page"}>Create account</a>
            </nav>
            ${isLogin
              ? html`<${LoginForm} onSignedIn=${onSignedIn} notice=${notice} />`
              : html`<${RegisterForm} onSignedIn=${onSignedIn} />`}
          </div>
        </main>
      </div>
    `;
  }

  // -------------------------------------------------------------------- home

  function Home({ session, onSignOut }) {
    useEffect(() => {
      document.title = "JobHunt";
    }, []);

    const tiles = [
      { href: "/docs", title: "My CV and matches", body: "Upload your CV and get job matches. Available in the API explorer for now.", cta: "Open API explorer" },
      { href: "/chatbot", title: "Profile assistant", body: "Ask questions about the candidate profile and resume.", cta: "Open assistant" },
      { href: "/upload", title: "Post a job", body: "Paste a raw job description to add it to the shared job pool.", cta: "Post a job" },
    ];

    return html`
      <div className="home">
        <header className="topbar">
          <div className="logo">Job<span>Hunt</span></div>
          <div className="who">
            <div className="avatar" aria-hidden="true">${session.username.charAt(0).toUpperCase()}</div>
            <span>${session.username}</span>
            <button type="button" className="btn secondary" onClick=${onSignOut}>Sign out</button>
          </div>
        </header>
        <div>
          <h1>Hi, ${session.username}</h1>
          <p className="lead">You're signed in. Pick where to go next.</p>
        </div>
        <div className="tiles">
          ${tiles.map(
            (t) => html`<a className="tile" href=${t.href} key=${t.href}>
              <strong>${t.title}</strong>
              <span>${t.body}</span>
              <span className="go">${t.cta} →</span>
            </a>`
          )}
        </div>
      </div>
    `;
  }

  // --------------------------------------------------------------------- app

  function App() {
    const route = useHashRoute();
    const [session, setSession] = useState(loadSession);
    const [notice, setNotice] = useState("");

    // Expire the session in place when the JWT runs out.
    useEffect(() => {
      if (!session) return undefined;
      const ms = session.expiresAt * 1000 - Date.now();
      const timer = setTimeout(() => {
        clearSession();
        setSession(null);
        setNotice("Your session expired. Sign in again.");
        go("login");
      }, Math.max(ms, 0));
      return () => clearTimeout(timer);
    }, [session]);

    // Signed-in users skip the auth pages; signed-out users can't see home.
    useEffect(() => {
      if (session && route !== "home") go("home");
      if (!session && route !== "login" && route !== "register") go("login");
    }, [route, session]);

    function onSignedIn(next) {
      setNotice("");
      setSession(next);
      go("home");
    }

    function onSignOut() {
      clearSession();
      setSession(null);
      setNotice("You've signed out.");
      go("login");
    }

    if (session) return html`<${Home} session=${session} onSignOut=${onSignOut} />`;
    return html`<${AuthPage} mode=${route === "register" ? "register" : "login"} onSignedIn=${onSignedIn} notice=${notice} />`;
  }

  ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
})();
