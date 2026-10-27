/*
 * Signed-in pages of the JobHunt web app: Home (choose CV / Jobs / Chatbot),
 * My CV, and Jobs (recommended + all open jobs). Every request goes through JH.api, which sends the session's
 * JWT -- the user is never asked for a token or API key.
 */
(function () {
  const { useState, useEffect } = React;
  const { html, api, ApiError, Field, TextArea, Switch, Segmented, SubmitButton, Alert, Loading, Chips, ScoreBadge, pretty } = window.JH;

  const MIN_MATCH = 45; // percent: jobs below this are never shown as matches

  function PageHeader({ title, lead, children }) {
    return html`
      <header className="page-head">
        <div>
          <h1>${title}</h1>
          ${lead && html`<p className="lead">${lead}</p>`}
        </div>
        ${children}
      </header>
    `;
  }

  function experienceLabel(min, max) {
    if (min == null && max == null) return null;
    if (min != null && max != null) return min === max ? `${min} yrs` : `${min}–${max} yrs`;
    return min != null ? `${min}+ yrs` : `up to ${max} yrs`;
  }

  function jobFacts(job) {
    return [
      job.company_name,
      job.location && job.location.length ? job.location.join(", ") : null,
      job.remote_type ? pretty(job.remote_type) : null,
      job.employment_type ? pretty(job.employment_type) : null,
      experienceLabel(job.experience_min, job.experience_max),
    ].filter(Boolean);
  }

  function formatDate(value) {
    if (!value) return null;
    const d = new Date(value);
    return isNaN(d) ? null : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  }

  function NoCvNotice() {
    return html`
      <div className="empty">
        <h2>Add your CV first</h2>
        <p>Matches and JD checks compare jobs against your CV.</p>
        <a className="btn" href="#/cv">Upload your CV</a>
      </div>
    `;
  }

  function Explanation({ explanation }) {
    if (!explanation) {
      return html`<p className="muted">The AI explanation isn't available right now. The scores and skills above are still accurate.</p>`;
    }
    return html`
      <div className="explanation">
        ${explanation.summary && html`<p>${explanation.summary}</p>`}
        <div className="two-col">
          ${explanation.supporting_points && explanation.supporting_points.length > 0 &&
          html`<div>
            <h4>Evidence from your CV</h4>
            <ul>${explanation.supporting_points.map((p, i) => html`<li key=${i}>${p}</li>`)}</ul>
          </div>`}
          ${explanation.gaps && explanation.gaps.length > 0 &&
          html`<div>
            <h4>Gaps</h4>
            <ul>${explanation.gaps.map((g, i) => html`<li key=${i}>${g}</li>`)}</ul>
          </div>`}
        </div>
      </div>
    `;
  }

  // ------------------------------------------------------------------ My CV

  const CV_MODES = [
    { value: "file", label: "Upload file" },
    { value: "text", label: "Paste text" },
    { value: "link", label: "From link" },
  ];
  const MAX_FILE_MB = 5;

  const SOURCE_LABEL = { file: "Uploaded file", text: "Pasted text", url: "Link" };

  function ChatbotCard({ hasCv }) {
    const [path, setPath] = useState(null);
    const [copied, setCopied] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
      api("/me").then((me) => setPath(me.chatbot_path)).catch((err) => setError(err.message));
    }, []);

    if (!path) return error ? html`<${Alert}>${error}<//>` : null;
    const link = window.location.origin + path;

    async function copy() {
      try {
        await navigator.clipboard.writeText(link);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (_) {
        setError("Couldn't copy. Select the link and copy it manually.");
      }
    }

    return html`
      <section className="card" aria-label="Your chatbot">
        <div>
          <h2>Your chatbot</h2>
          <p className="muted">Anyone with this link can chat with an AI that answers in your voice, only from your CV.</p>
        </div>
        <div className="share">
          <a href=${path} target="_blank" rel="noopener">${link}</a>
          <button type="button" className="btn secondary small" onClick=${copy}>${copied ? "Copied" : "Copy link"}</button>
        </div>
        ${!hasCv && html`<p className="muted">Add your CV below so the chatbot has something to answer from.</p>`}
        <${Alert}>${error}<//>
      </section>
    `;
  }

  function CvPage() {
    const [profile, setProfile] = useState(undefined); // undefined = loading, null = no CV yet
    const [loadError, setLoadError] = useState("");
    const [mode, setMode] = useState("file");
    const [file, setFile] = useState(null);
    const [text, setText] = useState("");
    const [link, setLink] = useState("");
    const [dragging, setDragging] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

    async function load() {
      try {
        setProfile(await api("/jobs/resume/me"));
      } catch (err) {
        if (err.status !== 404) setLoadError(err.message);
        setProfile(null);
      }
    }
    useEffect(() => {
      document.title = "My CV — JobHunt";
      load();
    }, []);

    function pickFile(f) {
      setError("");
      if (!f) return;
      if (!/\.(pdf|docx|txt)$/i.test(f.name)) return setError("Use a PDF, DOCX or TXT file.");
      if (f.size > MAX_FILE_MB * 1024 * 1024) return setError(`That file is over ${MAX_FILE_MB} MB.`);
      setFile(f);
    }

    async function submit(e) {
      e.preventDefault();
      setError("");
      setSuccess("");
      setBusy(true);
      try {
        if (mode === "file") {
          if (!file) throw new ApiError(0, "Choose a PDF, DOCX or TXT file first.");
          const form = new FormData();
          form.append("file", file);
          await api("/jobs/resume/upload", { form });
        } else if (mode === "text") {
          if (text.trim().length < 100) throw new ApiError(0, "Paste your whole CV. That looks too short.");
          await api("/jobs/resume", { json: { resume_text: text } });
        } else {
          if (!/^https?:\/\/\S+$/.test(link.trim())) throw new ApiError(0, "Enter a full link starting with https://");
          await api("/jobs/resume", { json: { resume_url: link.trim() } });
        }
        const replaced = Boolean(profile);
        setFile(null);
        setText("");
        setLink("");
        await load();
        setSuccess(replaced ? "Your CV was replaced. Matches now use the new version." : "Your CV is saved. Your job matches are ready.");
      } catch (err) {
        setError(err.message);
      } finally {
        setBusy(false);
      }
    }

    if (profile === undefined) return html`<${Loading} label="Loading your CV…" />`;

    return html`
      <${PageHeader}
        title="My CV"
        lead="One CV per account. Uploading a new one replaces the old CV everywhere, including your matches and the chatbot."
      >
        ${profile && html`<a className="btn secondary" href="#/jobs">See matching jobs</a>`}
      <//>
      <${Alert}>${loadError}<//>
      <${Alert} kind="success">${success}<//>

      ${profile
        ? html`<section className="card" aria-label="Current CV">
            <div className="card-row">
              <span className="pill ok">Active CV</span>
              ${formatDate(profile.updated_at) && html`<span className="muted">Updated ${formatDate(profile.updated_at)}</span>`}
              ${profile.resume_source &&
              html`<span className="muted">· ${SOURCE_LABEL[profile.resume_source] || profile.resume_source}${profile.resume_filename ? `: ${profile.resume_filename}` : ""}</span>`}
            </div>
            <div className="stats">
              <div><span>Skills found</span><strong>${profile.skills.length}</strong></div>
              <div><span>Experience</span><strong>${profile.experience != null ? profile.experience + " yrs" : "—"}</strong></div>
              <div><span>Industry</span><strong className="cap">${profile.industry_type ? pretty(profile.industry_type) : "—"}</strong></div>
              <div><span>Sections</span><strong>${profile.sections.length}</strong></div>
            </div>
            <h3>Skills detected</h3>
            <${Chips} items=${profile.skills} empty="No skills detected" />
            <h3>Sections we read</h3>
            <${Chips} items=${profile.sections} kind="outline" empty="No sections detected" />
          </section>`
        : html`<section className="card empty-inline"><strong>No CV yet.</strong> Add it below to start getting matches.</section>`}

      <${ChatbotCard} hasCv=${Boolean(profile)} />

      <form className="card" onSubmit=${submit} noValidate aria-label=${profile ? "Replace CV" : "Add CV"}>
        <div className="card-row spread">
          <h2>${profile ? "Replace your CV" : "Add your CV"}</h2>
          <${Segmented} label="How to add your CV" options=${CV_MODES} value=${mode} onChange=${(m) => { setMode(m); setError(""); }} />
        </div>

        ${mode === "file" &&
        html`<label
          className=${"dropzone" + (dragging ? " over" : "") + (file ? " has-file" : "")}
          onDragOver=${(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave=${() => setDragging(false)}
          onDrop=${(e) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
        >
          <svg aria-hidden="true" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"></path><path d="M7 9l5-5 5 5"></path><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"></path></svg>
          <strong>${file ? file.name : "Drop your CV here or choose a file"}</strong>
          <span className="muted">${file ? `${(file.size / 1024).toFixed(0)} KB · ready to upload` : `PDF, DOCX or TXT · up to ${MAX_FILE_MB} MB`}</span>
          <input type="file" accept=".pdf,.docx,.txt" onChange=${(e) => pickFile(e.target.files[0])} />
        </label>`}
        ${mode === "text" &&
        html`<${TextArea}
          id="cv-text"
          label="CV text"
          rows=${12}
          value=${text}
          onChange=${setText}
          placeholder="Paste the full text of your CV. Headings like Summary, Experience, Skills and Education are detected automatically."
        />`}
        ${mode === "link" &&
        html`<${Field}
          id="cv-link"
          label="Public link to your CV"
          type="url"
          placeholder="https://… (PDF, DOCX or plain text)"
          value=${link}
          onChange=${setLink}
        />`}

        ${profile && html`<div className="alert warn">Saving replaces your current CV. Its stored sections and search data are deleted and rebuilt from the new one.</div>`}
        <${Alert}>${error}<//>
        <div className="actions">
          <${SubmitButton} busy=${busy} busyLabel="Reading your CV… this can take up to a minute">
            ${profile ? "Replace CV" : "Save CV"}
          <//>
        </div>
      </form>
    `;
  }

  // ----------------------------------------------------------- Job matches

  function MatchCard({ rec, showGaps }) {
    const [open, setOpen] = useState(false);
    const [explanation, setExplanation] = useState(undefined); // undefined = not loaded
    const [error, setError] = useState("");
    const job = rec.job;

    async function toggle() {
      const next = !open;
      setOpen(next);
      if (next && explanation === undefined) {
        try {
          const body = await api("/jobs/explainMatch", { json: { job_id: job.id } });
          setExplanation(body.explanation);
        } catch (err) {
          setError(err.message);
          setExplanation(null);
        }
      }
    }

    const missing = [...rec.missing_required_skills, ...rec.missing_preferred_skills];
    return html`
      <article className="card match">
        <div className="match-top">
          <${ScoreBadge} score=${rec.fused_score} />
          <div className="match-main">
            <h2>${job.title}</h2>
            <p className="muted">${jobFacts(job).join(" · ")}</p>
            <p className="breakdown">
              ${rec.has_skill_data
                ? html`Skill match <strong>${Math.round(rec.skill_score * 100)}%</strong> · Profile similarity <strong>${Math.round(rec.vector_score * 100)}%</strong>`
                : html`No skills listed in this posting. Score is based on profile similarity only.`}
            </p>
          </div>
        </div>
        ${showGaps &&
        rec.has_skill_data &&
        html`<div className="two-col">
          <div><h4>You have</h4><${Chips} items=${[...rec.matched_required_skills, ...rec.matched_preferred_skills]} kind="have" /></div>
          <div><h4>Gaps</h4><${Chips} items=${missing} kind="gap" empty="No gaps. You have every listed skill." /></div>
        </div>`}
        <div className="match-foot">
          <button type="button" className="link-btn" aria-expanded=${open ? "true" : "false"} onClick=${toggle}>
            ${open ? "Hide explanation" : "Why this match?"}
          </button>
          ${open &&
          (explanation === undefined
            ? html`<${Loading} label="Asking the AI to explain this match…" />`
            : html`<div><${Alert}>${error}<//><${Explanation} explanation=${explanation} /></div>`)}
        </div>
      </article>
    `;
  }

  const JOB_PAGE_SIZE = 50;

  function JobCard({ job, match }) {
    return html`
      <article className="card match">
        <div className="match-top">
          ${match ? html`<${ScoreBadge} score=${match.fused_score} />` : null}
          <div className="match-main">
            <div className="card-row">
              <h2>${job.title}</h2>
              ${match && html`<span className="pill ok">Recommended</span>`}
            </div>
            <p className="muted">${jobFacts(job).join(" · ")}</p>
          </div>
        </div>
        <div><h4>Required skills</h4><${Chips} items=${job.required_skills} empty="None listed" /></div>
        ${job.preferred_skills.length > 0 && html`<div><h4>Nice to have</h4><${Chips} items=${job.preferred_skills} kind="outline" /></div>`}
        ${match && html`<a className="link-btn" href="#/jobs" onClick=${(e) => { e.preventDefault(); window.dispatchEvent(new CustomEvent("jobhunt:show-recommended")); }}>See why it matches →</a>`}
      </article>
    `;
  }

  function JobsPage() {
    const [tab, setTab] = useState("recommended");
    const [recs, setRecs] = useState(undefined); // undefined = loading, null = no CV yet
    const [jobs, setJobs] = useState(undefined);
    const [total, setTotal] = useState(0);
    const [loadingMore, setLoadingMore] = useState(false);
    const [error, setError] = useState("");
    const [minScore, setMinScore] = useState(MIN_MATCH);
    const [remoteOnly, setRemoteOnly] = useState(false);
    const [showGaps, setShowGaps] = useState(true);
    const [sort, setSort] = useState("best");
    const [query, setQuery] = useState("");

    useEffect(() => {
      document.title = "Jobs — JobHunt";
      api(`/jobs/recommendations?limit=50&min_score=${MIN_MATCH / 100}`)
        .then((body) => setRecs(body.recommendations))
        .catch((err) => {
          if (err.status !== 404) setError(err.message);
          setRecs(null);
        });
      api(`/jobs?status=ACTIVE&limit=${JOB_PAGE_SIZE}`)
        .then((body) => {
          setJobs(body.jobs);
          setTotal(body.total);
        })
        .catch((err) => {
          setError(err.message);
          setJobs([]);
        });
      const showRecommended = () => setTab("recommended");
      window.addEventListener("jobhunt:show-recommended", showRecommended);
      return () => window.removeEventListener("jobhunt:show-recommended", showRecommended);
    }, []);

    async function loadMore() {
      setLoadingMore(true);
      try {
        const body = await api(`/jobs?status=ACTIVE&limit=${JOB_PAGE_SIZE}&offset=${jobs.length}`);
        setJobs((prev) => [...prev, ...body.jobs]);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingMore(false);
      }
    }

    if (recs === undefined || jobs === undefined) return html`<${Loading} label="Loading jobs and scoring them against your CV…" />`;

    const matchById = new Map((recs || []).map((r) => [r.job.id, r]));
    const missingCount = (r) => r.missing_required_skills.length + r.missing_preferred_skills.length;
    const recommended = (recs || [])
      .filter((r) => r.fused_score * 100 >= minScore && (!remoteOnly || r.job.remote_type === "remote"))
      .sort((a, b) => (sort === "gaps" ? missingCount(a) - missingCount(b) : 0) || b.fused_score - a.fused_score);

    const q = query.trim().toLowerCase();
    const openJobs = jobs.filter(
      (j) =>
        !q ||
        [j.title, j.company_name, ...(j.location || []), ...j.required_skills, ...j.preferred_skills].some((v) =>
          String(v || "").toLowerCase().includes(q)
        )
    );

    const tabs = [
      { value: "recommended", label: `Recommended for you${recs ? ` (${recs.length})` : ""}` },
      { value: "all", label: `All open jobs (${total})` },
    ];

    return html`
      <${PageHeader} title="Jobs" lead=${`Open jobs from the shared pool. Jobs your CV matches at ${MIN_MATCH}% or higher are recommended to you.`}>
        <${Segmented} label="Which jobs to show" options=${tabs} value=${tab} onChange=${setTab} />
      <//>
      <${Alert}>${error}<//>

      ${tab === "recommended" &&
      (recs === null
        ? html`<${NoCvNotice} />`
        : html`
            <section className="card filters" aria-label="Filters">
              <div className="field range">
                <label htmlFor="min-score">Minimum match <strong>${minScore}%</strong></label>
                <input id="min-score" type="range" min=${MIN_MATCH} max="90" step="5" value=${minScore} onChange=${(e) => setMinScore(Number(e.target.value))} />
              </div>
              <${Switch} id="remote-only" label="Remote only" checked=${remoteOnly} onChange=${setRemoteOnly} />
              <${Switch} id="show-gaps" label="Show skill gaps" checked=${showGaps} onChange=${setShowGaps} />
              <div className="field">
                <label htmlFor="sort">Sort by</label>
                <select id="sort" value=${sort} onChange=${(e) => setSort(e.target.value)}>
                  <option value="best">Best match</option>
                  <option value="gaps">Fewest skill gaps</option>
                </select>
              </div>
            </section>
            <p className="muted count">${recommended.length === 1 ? "1 job" : `${recommended.length} jobs`} at ${minScore}% or higher</p>
            ${recommended.length === 0
              ? html`<div className="empty">
                  <h2>No recommended jobs at this level yet</h2>
                  <p>Lower the minimum match or browse all open jobs. New jobs are scored against your CV automatically.</p>
                  <button type="button" className="btn secondary" onClick=${() => setTab("all")}>Browse all open jobs</button>
                </div>`
              : recommended.map((r) => html`<${MatchCard} key=${r.job.id} rec=${r} showGaps=${showGaps} />`)}
          `)}

      ${tab === "all" &&
      html`
        <div className="field search">
          <label htmlFor="job-search" className="sr-only">Search jobs</label>
          <input id="job-search" type="search" placeholder="Search by title, company, location or skill" value=${query} onChange=${(e) => setQuery(e.target.value)} />
        </div>
        ${recs === null && html`<div className="alert warn">Add your CV to see which of these jobs match you. <a href="#/cv">Upload your CV</a></div>`}
        <p className="muted count">
          ${q ? `${openJobs.length} of ${jobs.length} loaded jobs match "${query.trim()}"` : `Showing ${jobs.length} of ${total} open jobs`}
        </p>
        ${openJobs.length === 0
          ? html`<div className="empty"><h2>No open jobs found</h2><p>${q ? "Try a different search." : "No jobs have been posted yet."}</p></div>`
          : openJobs.map((j) => html`<${JobCard} key=${j.id} job=${j} match=${matchById.get(j.id)} />`)}
        ${jobs.length < total &&
        html`<div className="actions start">
          <button type="button" className="btn secondary" onClick=${loadMore} disabled=${loadingMore}>
            ${loadingMore ? "Loading…" : "Load more jobs"}
          </button>
        </div>`}
      `}
    `;
  }

  // ------------------------------------------------------------------- Home

  function HomePage({ session }) {
    const [status, setStatus] = useState(null);

    useEffect(() => {
      document.title = "Home — JobHunt";
      const optional = (promise) => promise.catch(() => null);
      Promise.all([
        optional(api("/jobs/resume/me")),
        optional(api(`/jobs/recommendations?limit=50&min_score=${MIN_MATCH / 100}`)),
        optional(api("/me")),
      ]).then(([cv, recs, account]) => setStatus({ cv, recs: recs ? recs.recommendations : null, account }));
    }, []);

    const name = (status && status.account && status.account.full_name) || session.username;
    const cv = status && status.cv;
    const recCount = status && status.recs ? status.recs.length : 0;
    // Unknown until /me answers -- never guess, or a quick click could open someone else's bot.
    const chatbotPath = status && status.account ? status.account.chatbot_path : null;

    const tiles = [
      {
        key: "cv",
        href: "#/cv",
        title: "My CV",
        body: "Upload your CV as a file, pasted text or a link. A new upload replaces the old one everywhere.",
        state: !status ? "…" : cv ? `Uploaded · ${cv.skills.length} skills found` : "No CV yet",
        ok: Boolean(cv),
        cta: cv ? "View or replace CV" : "Add your CV",
      },
      {
        key: "jobs",
        href: "#/jobs",
        title: "Jobs",
        body: `Browse all open jobs. Jobs your CV matches at ${MIN_MATCH}% or higher are recommended to you.`,
        state: !status ? "…" : !cv ? "Add your CV to get recommendations" : `${recCount} recommended for you`,
        ok: recCount > 0,
        cta: "See jobs",
      },
      {
        key: "chatbot",
        href: chatbotPath || "#/home",
        external: Boolean(chatbotPath),
        title: "Chatbot",
        body: "An AI that answers recruiters' questions in your voice, only from your CV. Share the link with anyone.",
        state: !status ? "…" : cv ? "Ready · answers from your CV" : "Add your CV so it can answer",
        ok: Boolean(cv),
        cta: "Open my chatbot",
      },
    ];

    return html`
      <${PageHeader} title=${`Hi, ${name.split(" ")[0]}`} lead="What would you like to do?" />
      <div className="tiles">
        ${tiles.map(
          (t) => html`<a
            key=${t.key}
            className="tile"
            href=${t.href}
            target=${t.external ? "_blank" : undefined}
            rel=${t.external ? "noopener" : undefined}
          >
            <strong>${t.title}</strong>
            <span className="tile-body">${t.body}</span>
            <span className=${"tile-state" + (t.ok ? " ok" : "")}>${t.state}</span>
            <span className="go">${t.cta} →</span>
          </a>`
        )}
      </div>
    `;
  }

  window.JH.pages = { HomePage, CvPage, JobsPage };
})();
