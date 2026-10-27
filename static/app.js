/*
 * JobHunt web app shell: sign in / create account, then the signed-in app
 * (Home, My CV, Jobs; the chatbot is its own page at /chatbot). React 18 via UMD + htm for
 * JSX-like templates -- no build step, served by FastAPI from /static.
 * Shared pieces live in core.js, the signed-in pages in pages.js.
 *
 * Routes are hash-based (#/login, #/register, #/home, #/cv, #/jobs) so they never collide
 * with the POST /login and POST /register API routes.
 */
(function () {
  const { useState, useEffect, useCallback } = React;
  const { html, api, login, loadSession, clearSession, Field, Switch, SubmitButton, Alert, ThemeToggle, pages } = window.JH;

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
        await api("/register", { auth: false, json: {
          username: form.username.trim(),
          password: form.password,
          full_name: optional(form.full_name),
          email: optional(form.email),
          age: Number(form.age),
          graduated: form.graduated,
          linkedin_url: showLinks ? optional(form.linkedin_url) : null,
          leetcode_url: showLinks ? optional(form.leetcode_url) : null,
          hackerrank_url: showLinks ? optional(form.hackerrank_url) : null,
        } });
        // Account created -- sign straight in.
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
          <${ThemeToggle} />
          <div className="auth-card">
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

  // ------------------------------------------------------- signed-in shell

  // `hidden` routes are reachable (the logo goes Home) but not shown as tabs.
  const NAV = [
    { route: "home", label: "Home", page: pages.HomePage, hidden: true },
    { route: "cv", label: "My CV", page: pages.CvPage },
    { route: "jobs", label: "Jobs", page: pages.JobsPage },
  ];
  // Signing in lands on Home, where the user picks My CV, Jobs or Chatbot.
  const DEFAULT_ROUTE = "home";

  function Shell({ route, session, onSignOut }) {
    const current = NAV.find((n) => n.route === route) || NAV.find((n) => n.route === DEFAULT_ROUTE);
    const Page = current.page;
    return html`
      <div className="shell">
        <header className="appbar">
          <a className="logo" href="#/home">Job<span>Hunt</span></a>
          <nav className="appnav" aria-label="Main">
            ${NAV.filter((n) => !n.hidden).map(
              (n) => html`<a key=${n.route} href=${"#/" + n.route} aria-current=${n === current ? "page" : undefined}>${n.label}</a>`
            )}
          </nav>
          <div className="who">
            <${ThemeToggle} />
            <div className="avatar" aria-hidden="true">${session.username.charAt(0).toUpperCase()}</div>
            <span className="who-name">${session.username}</span>
            <button type="button" className="btn secondary small" onClick=${onSignOut}>Sign out</button>
          </div>
        </header>
        <main className="page" id="main">
          <${Page} key=${current.route} session=${session} />
        </main>
      </div>
    `;
  }

  // --------------------------------------------------------------------- app

  const AUTH_ROUTES = ["login", "register"];

  function App() {
    const route = useHashRoute();
    const [session, setSession] = useState(loadSession);
    const [notice, setNotice] = useState("");

    const endSession = useCallback((message) => {
      clearSession();
      setSession(null);
      setNotice(message);
      go("login");
    }, []);

    // Expire the session in place when the JWT runs out, or when the API says it's no longer valid.
    useEffect(() => {
      if (!session) return undefined;
      const timer = setTimeout(() => endSession("Your session expired. Sign in again."), Math.max(session.expiresAt * 1000 - Date.now(), 0));
      const onUnauthorized = () => endSession("Your session expired. Sign in again.");
      window.addEventListener("jobhunt:unauthorized", onUnauthorized);
      return () => {
        clearTimeout(timer);
        window.removeEventListener("jobhunt:unauthorized", onUnauthorized);
      };
    }, [session, endSession]);

    // Signed-in users skip the auth pages; signed-out users only see them.
    useEffect(() => {
      if (session && (AUTH_ROUTES.includes(route) || !NAV.some((n) => n.route === route))) go(DEFAULT_ROUTE);
      if (!session && !AUTH_ROUTES.includes(route)) go("login");
    }, [route, session]);

    function onSignedIn(next, landing = DEFAULT_ROUTE) {
      setNotice("");
      setSession(next);
      go(landing);
    }

    if (session) return html`<${Shell} route=${route} session=${session} onSignOut=${() => endSession("You've signed out.")} />`;
    return html`<${AuthPage} mode=${route === "register" ? "register" : "login"} onSignedIn=${onSignedIn} notice=${notice} />`;
  }

  ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
})();
