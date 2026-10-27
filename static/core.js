/*
 * Shared building blocks for the JobHunt web app (loaded before pages.js and
 * app.js): session storage, the API client, and small UI primitives.
 * Everything is exposed on window.JH -- there's no bundler.
 */
(function () {
  const html = htm.bind(React.createElement);
  const { useState } = React;

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

  // FastAPI returns {detail: "msg"} for HTTPException, {detail: [{loc, msg}]}
  // for validation (422) errors, and the auth middleware {message: "..."}.
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

  /**
   * Call the API as the signed-in user: the session's JWT is attached
   * automatically, so no page ever asks for a token. A 401 means the session
   * is gone; the app listens for "jobhunt:unauthorized" and returns to sign-in.
   *   api("/jobs/resume/me")
   *   api("/jobs/resume", { json: {...} })
   *   api("/jobs/resume/upload", { form: formData })
   */
  async function api(path, { method, json, form, auth = true } = {}) {
    const headers = {};
    if (json !== undefined) headers["Content-Type"] = "application/json";
    const session = auth ? loadSession() : null;
    if (session) headers.Authorization = "Bearer " + session.jwt;

    let response;
    try {
      response = await fetch(path, {
        method: method || (json !== undefined || form ? "POST" : "GET"),
        headers,
        body: json !== undefined ? JSON.stringify(json) : form,
      });
    } catch (_) {
      throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
    }
    let body = null;
    try {
      body = await response.json();
    } catch (_) {}
    if (response.status === 401 && auth) window.dispatchEvent(new Event("jobhunt:unauthorized"));
    if (!response.ok) throw toApiError(response.status, body);
    return body;
  }

  async function login(username, password) {
    const body = await api("/login", { json: { username, password }, auth: false });
    const session = { jwt: body.jwt, username: username.trim().toLowerCase(), expiresAt: body.jwt_expires_at };
    saveSession(session);
    return session;
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

  function TextArea({ id, label, hint, value, onChange, rows = 8, ...rest }) {
    return html`
      <div className="field">
        <label htmlFor=${id}>${label}${hint ? html` <span className="hint">${hint}</span>` : null}</label>
        <textarea id=${id} rows=${rows} value=${value} onChange=${(e) => onChange(e.target.value)} ...${rest}></textarea>
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

  /** Pill-style single choice, e.g. Upload file / Paste text / From link. */
  function Segmented({ label, options, value, onChange }) {
    return html`
      <div className="segmented" role="radiogroup" aria-label=${label}>
        ${options.map(
          (o) => html`<button
            key=${o.value}
            type="button"
            role="radio"
            aria-checked=${value === o.value ? "true" : "false"}
            onClick=${() => onChange(o.value)}
          >
            ${o.label}
          </button>`
        )}
      </div>
    `;
  }

  function SubmitButton({ busy, children, busyLabel }) {
    return html`
      <button type="submit" className="btn" disabled=${busy} aria-busy=${busy ? "true" : "false"}>
        ${busy && html`<span className="spinner" aria-hidden="true"></span>`}
        ${busy && busyLabel ? busyLabel : children}
      </button>
    `;
  }

  function Alert({ kind = "error", children }) {
    if (!children) return null;
    return html`<div className=${"alert " + kind} role=${kind === "error" ? "alert" : "status"}>${children}</div>`;
  }

  function Loading({ label = "Loading…" }) {
    return html`<div className="loading" role="status"><span className="spinner dark" aria-hidden="true"></span>${label}</div>`;
  }

  function Chips({ items, kind = "", empty = "None" }) {
    if (!items || !items.length) return html`<span className="muted">${empty}</span>`;
    return html`<div className="chips">${items.map((s) => html`<span key=${s} className=${"chip " + kind}>${pretty(s)}</span>`)}</div>`;
  }

  /** Match bands: Strong >= 70%, Good >= 55%, Fair >= 45%, else Low. */
  function band(score) {
    const pct = Math.round(score * 100);
    if (pct >= 70) return { pct, label: "Strong", kind: "strong" };
    if (pct >= 55) return { pct, label: "Good", kind: "good" };
    if (pct >= 45) return { pct, label: "Fair", kind: "fair" };
    return { pct, label: "Low", kind: "low" };
  }

  function ScoreBadge({ score, size = "md" }) {
    const b = band(score);
    return html`
      <div className=${`score ${b.kind} ${size}`} aria-label=${`${b.pct}% match, ${b.label}`}>
        <span className="pct">${b.pct}%</span>
        <span className="lbl">${b.label}</span>
      </div>
    `;
  }

  const THEME_KEY = "jobhunt_theme";

  /** Day/night switch. The theme lives on <html data-theme>, set before first paint by index.html. */
  function ThemeToggle() {
    const [theme, setTheme] = useState(() => document.documentElement.getAttribute("data-theme") || "dark");
    function flip() {
      const next = theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch (_) {}
      setTheme(next);
    }
    const toLight = theme === "dark";
    return html`
      <button type="button" className="theme-toggle" onClick=${flip} aria-label=${toLight ? "Switch to day mode" : "Switch to night mode"} title=${toLight ? "Day mode" : "Night mode"}>
        ${toLight
          ? html`<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>`
          : html`<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>`}
      </button>
    `;
  }

  // skill ids like "spring_boot" -> "spring boot"
  function pretty(text) {
    return String(text).replace(/_/g, " ");
  }

  window.JH = {
    html,
    loadSession,
    saveSession,
    clearSession,
    api,
    login,
    ApiError,
    Field,
    TextArea,
    Switch,
    Segmented,
    SubmitButton,
    Alert,
    Loading,
    Chips,
    ScoreBadge,
    ThemeToggle,
    band,
    pretty,
  };
})();
