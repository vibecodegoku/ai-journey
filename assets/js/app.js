document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  if (!store.get('user.joinedDate')) {
    store.set('user.joinedDate', new Date().toISOString());
  }

  const theme = store.get('user.theme') || 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  updateThemeIcon(theme);

  updateStreak();
  updateHeaderStats();

  // ── Cloud sync: init Supabase, then handle session ───────────────
  const syncEnabled = (typeof supabaseSync !== 'undefined') && supabaseSync.init();
  if (syncEnabled) {
    await initAuth();
  } else {
    // Hide sync button when Supabase is not configured
    const btn = document.getElementById('cloud-sync-btn');
    if (btn) btn.style.display = 'none';
  }
  // ─────────────────────────────────────────────────────────────────

  if (!store.get('user.name')) {
    showWelcomeModal(syncEnabled);
  }

  renderTimeline();
  renderProjects();
  renderCertifications();
  renderCourses();
  renderChannels();
  renderBadges();
  renderSystemDesign();

  initFilters();
  restoreFilters();
  initModals();
  initThemeToggle();
  initProfileEdit();
  initExportImport();

  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
  });
}

// ============================================================
//  Auth — Supabase / GitHub OAuth
// ============================================================

async function initAuth() {
  const btn = document.getElementById('cloud-sync-btn');
  if (!btn) return;

  // Register auth state listener — fires on sign-in, sign-out, token refresh
  supabaseSync.onAuthChange(async (event, user) => {
    updateAuthButton(user);

    if (event === 'SIGNED_IN' && user) {
      const synced = await loadCloudState();
      // Auto-fill name from GitHub if not yet set
      if (!store.get('user.name')) {
        const ghName = supabaseSync.getUserDisplayName();
        if (ghName) store.set('user.name', ghName);
        updateHeaderStats();
        // Close welcome modal if open
        const wm = document.getElementById('welcome-modal');
        if (wm) wm.remove();
      }
      showToast(synced ? '☁️ Progress loaded from cloud!' : '☁️ Signed in — progress will sync across devices.', 'success');
    } else if (event === 'SIGNED_OUT') {
      updateAuthButton(null);
      showToast('Signed out — progress still saved locally.', 'info');
    }
  });

  // If a session already exists (returning user / post-OAuth redirect)
  if (supabaseSync.isReady()) {
    updateAuthButton(supabaseSync.getUser());
    await loadCloudState();
    // Auto-fill name from GitHub profile if missing
    if (!store.get('user.name')) {
      const ghName = supabaseSync.getUserDisplayName();
      if (ghName) { store.set('user.name', ghName); updateHeaderStats(); }
    }
  } else {
    updateAuthButton(null);
  }

  // Button click: sign in or sign out
  btn.addEventListener('click', async () => {
    if (supabaseSync.isReady()) {
      showSignOutConfirm();
    } else {
      btn.disabled = true;
      btn.textContent = 'Redirecting…';
      try {
        await supabaseSync.signInWithGitHub();
        // Page will redirect to GitHub — execution stops here
      } catch (err) {
        showToast('Sign-in failed: ' + err.message, 'error');
        updateAuthButton(null);
        btn.disabled = false;
      }
    }
  });
}

// Pull cloud state and merge: higher XP wins.
async function loadCloudState() {
  if (!supabaseSync.isReady()) return false;

  const cloudData = await supabaseSync.pullState();

  if (!cloudData?.state) {
    // No cloud record yet — push local progress up
    supabaseSync.schedulePush(JSON.parse(store.export()));
    return false;
  }

  const cloudXP = cloudData.state?.user?.xp || 0;
  const localXP = store.get('user.xp') || 0;

  if (cloudXP > localXP) {
    // Cloud is ahead — replace local state
    store.import(JSON.stringify(cloudData.state));
    renderAllSections();
    updateHeaderStats();
    return true;
  } else {
    // Local is ahead (or equal) — push local to cloud
    supabaseSync.schedulePush(JSON.parse(store.export()));
    return false;
  }
}

function renderAllSections() {
  renderTimeline();
  renderProjects();
  renderCertifications();
  renderCourses();
  renderChannels();
  renderBadges();
}

function updateAuthButton(user) {
  const btn = document.getElementById('cloud-sync-btn');
  const ind = document.getElementById('sync-indicator');
  if (!btn) return;

  if (user) {
    const avatar = supabaseSync.getUserAvatar();
    const name   = supabaseSync.getUserDisplayName();
    btn.innerHTML = `
      ${avatar ? `<img src="${avatar}" alt="" class="auth-avatar">` : '☁️ '}
      <span class="auth-name">${name}</span>
      <span class="auth-signout-hint">▾</span>
    `;
    btn.className = 'btn-auth btn-auth--signed-in';
    btn.title     = 'Signed in — click to sign out';
    if (ind) ind.style.display = '';
  } else {
    btn.innerHTML = '☁️ Sync Progress';
    btn.className = 'btn-auth btn-auth--signed-out';
    btn.title     = 'Sign in with GitHub to sync progress across devices';
    btn.disabled  = false;
    if (ind) { ind.textContent = ''; ind.style.display = 'none'; }
  }
}

function showSignOutConfirm() {
  const existing = document.getElementById('signout-modal');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'signout-modal';
  overlay.className = 'modal-overlay modal--open';
  overlay.style.zIndex = '700';
  overlay.innerHTML = `
    <div class="modal-box" style="max-width:380px;padding:2rem;text-align:center">
      <div style="font-size:2rem;margin-bottom:0.5rem">☁️</div>
      <h3 style="margin-bottom:0.5rem">Sign Out?</h3>
      <p style="margin-bottom:1.5rem;color:var(--text-secondary);font-size:.9rem">
        Your progress will remain saved locally. Sign in again on any device to restore it.
      </p>
      <div style="display:flex;gap:.75rem;justify-content:center">
        <button class="btn btn--outline" onclick="document.getElementById('signout-modal').remove()">Cancel</button>
        <button class="btn btn--primary" onclick="confirmSignOut()">Sign Out</button>
      </div>
    </div>
  `;
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
}

async function confirmSignOut() {
  const modal = document.getElementById('signout-modal');
  if (modal) modal.remove();
  await supabaseSync.signOut();
}

function renderTimeline() {
  const container = document.getElementById('timeline-nodes');
  if (!container) return;

  const months = [
    { num: 1, title: 'ML Foundations', sub: 'Python, scikit-learn, AWS SageMaker', emoji: '🌱' },
    { num: 2, title: 'Cloud LLMs', sub: 'Azure AI, AWS Bedrock, Foundation Models', emoji: '☁️' },
    { num: 3, title: 'LLMs & RAG', sub: 'Fine-tuning, Vector DBs, RAGAS', emoji: '🧠' },
    { num: 4, title: 'AI Agents', sub: 'Vertex AI, LangGraph, Tool Use', emoji: '🤖' },
    { num: 5, title: 'Multi-Agent', sub: 'CrewAI, MLOps, Kafka Streaming', emoji: '🎭' },
    { num: 6, title: 'Architecture', sub: 'Multi-Cloud, Governance, Capstone', emoji: '👑' }
  ];

  const completed = store.get('completed.projects');
  const projectsByMonth = {};
  DATA.projects.forEach(p => {
    if (!projectsByMonth[p.month]) projectsByMonth[p.month] = [];
    projectsByMonth[p.month].push(p);
  });

  container.innerHTML = months.map(m => {
    const monthProjects = projectsByMonth[m.num] || [];
    const completedCount = monthProjects.filter(p => completed.includes(p.id)).length;
    const isDone = completedCount === monthProjects.length && monthProjects.length > 0;
    const isActive = !isDone && completedCount > 0;

    return `
      <div class="timeline-node ${isDone ? 'timeline-node--done' : isActive ? 'timeline-node--active' : ''}"
           onclick="document.getElementById('section-projects').scrollIntoView({behavior:'smooth'})">
        <div class="node-marker">
          ${isDone ? '✓' : m.emoji}
        </div>
        <div class="node-label">Month ${m.num}</div>
        <div class="node-content">
          <strong>${m.title}</strong>
          <span>${m.sub}</span>
          <span class="node-progress">${completedCount}/${monthProjects.length} projects</span>
        </div>
      </div>
    `;
  }).join('');
}

function renderProjects() {
  const container = document.getElementById('projects-grid');
  if (!container) return;

  const completed = store.get('completed.projects');

  container.innerHTML = DATA.projects.map(p => {
    const isDone = completed.includes(p.id);
    const providerIcons = p.providers.map(pr => providerBadge(pr)).join('');
    const categoryTags = p.categories.map(c => `<span class="tag">${c.replace('-', ' ')}</span>`).join('');

    return `
      <div class="card card--project ${isDone ? 'card--completed' : ''}"
           data-filterable
           data-providers="${p.providers.join(',')}"
           data-cost="${p.cost}"
           data-categories="${p.categories.join(',')}">
        <div class="card__header">
          <div class="card__providers">${providerIcons}</div>
          <span class="card__xp ${isDone ? 'card__xp--earned' : ''}">
            ${isDone ? '✓ ' : '+'} ${p.xp} XP
          </span>
        </div>
        <div class="card__week">${p.week}</div>
        <h3 class="card__title">${p.num !== '00' ? `P${p.num}: ` : ''}${p.title}</h3>
        <p class="card__subtitle">${p.subtitle}</p>
        <p class="card__description">${p.description}</p>
        <div class="card__meta">
          <span class="meta-item">${p.cost === 'free' ? '✅ Free' : '💳 ' + p.estimatedCost}</span>
          <span class="meta-item">🛠 ${p.tools.slice(0, 3).join(', ')}</span>
        </div>
        <div class="card__tags">${categoryTags}</div>
        <div class="card__actions">
          <button class="btn btn--outline" onclick="openProjectModal('${p.id}')">
            📋 View Details
          </button>
          <button class="btn ${isDone ? 'btn--done' : 'btn--primary'}"
                  onclick="toggleProjectComplete('${p.id}')"
                  ${isDone ? '' : ''}>
            ${isDone ? '✓ Completed' : 'Mark Complete'}
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function renderCertifications() {
  const container = document.getElementById('certs-grid');
  if (!container) return;

  const completed = store.get('completed.certifications');
  const inProgress = store.get('inProgress.certifications');

  container.innerHTML = DATA.certifications.map(c => {
    const isDone = completed.includes(c.id);
    const isStarted = inProgress.includes(c.id);
    const stars = '★'.repeat(c.difficulty) + '☆'.repeat(5 - c.difficulty);

    return `
      <div class="card card--cert ${isDone ? 'card--completed' : ''}"
           data-filterable
           data-providers="${c.provider}"
           data-cost="${c.cost}"
           data-categories="${c.tags.join(',')}">
        <div class="card__header">
          <div class="cert-provider-logo">${providerLogo(c.provider)}</div>
          <div class="cert-badges">
            <span class="badge badge--${c.cost}">${c.price}</span>
            ${c.hotBadge ? `<span class="badge badge--hot">${c.hotBadge}</span>` : ''}
          </div>
        </div>
        <div class="cert-code">${c.code}</div>
        <h3 class="card__title">${c.title}</h3>
        <p class="card__description">${c.description}</p>
        <div class="card__meta">
          <span class="meta-item cert-stars">${stars}</span>
          <span class="meta-item">📅 ${c.examMonth}</span>
          ${c.duration !== 'Self-paced' && c.duration !== 'TBD' ? `<span class="meta-item">⏱ ${c.duration}</span>` : ''}
        </div>
        <div class="card__actions">
          <a href="${c.link}" target="_blank" rel="noopener" class="btn btn--outline">
            🔗 Official Page
          </a>
          ${!isDone ? `
            ${!isStarted ? `<button class="btn btn--secondary" onclick="startCert('${c.id}')">📖 Start Studying</button>` : ''}
            ${isStarted ? `<button class="btn btn--primary" onclick="passCert('${c.id}')">✅ Mark Passed</button>` : ''}
          ` : `<button class="btn btn--done">✓ Passed</button>`}
        </div>
      </div>
    `;
  }).join('');
}

function renderCourses() {
  const container = document.getElementById('courses-grid');
  if (!container) return;

  const completed = store.get('completed.courses');

  container.innerHTML = DATA.courses.map(c => {
    const isDone = completed.includes(c.id);
    return `
      <div class="card card--course ${isDone ? 'card--completed' : ''}"
           data-filterable
           data-providers="${c.provider}"
           data-cost="${c.cost}"
           data-categories="${c.tags.join(',')}">
        <div class="card__header">
          <div class="course-platform">${providerBadge(c.provider)}</div>
          <span class="badge badge--${c.cost}">${c.cost === 'free' ? 'Free' : c.cost}</span>
        </div>
        <h3 class="card__title">${c.title}</h3>
        <div class="course-platform-name">${c.platform}</div>
        <p class="card__description">${c.description}</p>
        <div class="card__meta">
          <span class="meta-item">⏱ ${c.duration}</span>
          <span class="meta-item">+${c.xp || 75} XP</span>
        </div>
        <div class="card__actions">
          <a href="${c.link}" target="_blank" rel="noopener" class="btn btn--outline">▶ Start Course</a>
          <button class="btn ${isDone ? 'btn--done' : 'btn--primary'}" onclick="completeCourse('${c.id}')">
            ${isDone ? '✓ Done' : 'Mark Done'}
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function renderChannels() {
  const container = document.getElementById('channels-grid');
  if (!container) return;

  const completed = store.get('completed.channels');

  container.innerHTML = DATA.channels.map(ch => {
    const isWatched = completed.includes(ch.id);
    return `
      <div class="card card--channel ${isWatched ? 'card--completed' : ''}"
           data-filterable
           data-providers="${ch.id.includes('gcp') ? 'gcp' : ch.id.includes('aws') ? 'aws' : 'general'}"
           data-cost="free"
           data-categories="${ch.tags.join(',')}">
        <div class="card__header">
          <div class="channel-icon">📺</div>
          <span class="badge badge--free">Free</span>
        </div>
        <h3 class="card__title">${ch.title}</h3>
        <div class="channel-subs">${ch.subscribers} subscribers</div>
        <p class="card__description">${ch.focus}</p>
        <div class="card__actions">
          <a href="${ch.link}" target="_blank" rel="noopener" class="btn btn--outline">▶ Watch</a>
          <button class="btn ${isWatched ? 'btn--done' : 'btn--primary'}" onclick="completeChannel('${ch.id}')">
            ${isWatched ? '✓ Watched' : '+25 XP'}
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function renderBadges() {
  const container = document.getElementById('badges-grid');
  if (!container) return;

  const earned = store.get('badges.earned');

  container.innerHTML = DATA.badges.map(b => {
    const isEarned = earned.includes(b.id);
    return `
      <div class="badge-card ${isEarned ? 'badge-card--earned' : 'badge-card--locked'}"
           title="${isEarned ? `Earned: ${b.description}` : `Locked: ${b.description}`}">
        <div class="badge-emoji">${isEarned ? b.emoji : '🔒'}</div>
        <div class="badge-name">${b.name}</div>
        <div class="badge-tier badge-tier--${b.tier}">${b.tier}</div>
        ${!isEarned ? `<div class="badge-lock-hint">${b.description}</div>` : ''}
      </div>
    `;
  }).join('');
}

function renderSystemDesign() {
  const container = document.getElementById('sysdesign-grid');
  if (!container) return;

  container.innerHTML = DATA.systemDesign.map(sd => {
    const diffColor = sd.difficulty === 'Hard' ? 'var(--accent-red)' : sd.difficulty === 'Medium' ? 'var(--accent-yellow)' : 'var(--accent-green)';
    return `
      <div class="card card--sysdesign"
           data-filterable
           data-providers="general"
           data-cost="free"
           data-categories="${sd.tags.join(',')}">
        <div class="card__header">
          <span class="sysdesign-icon">🏗</span>
          <span class="badge" style="background:${diffColor};color:#fff">${sd.difficulty}</span>
        </div>
        <h3 class="card__title">${sd.title}</h3>
        <div class="sysdesign-context">${sd.context}</div>
        <div class="sysdesign-components">
          <strong>Key Components:</strong>
          <div class="component-list">
            ${sd.keyComponents.map(c => `<span class="component-chip">${c}</span>`).join('')}
          </div>
        </div>
        <div class="sysdesign-tradeoffs">
          <strong>Trade-offs to discuss:</strong>
          <ul>${sd.tradeoffs.map(t => `<li>${t}</li>`).join('')}</ul>
        </div>
      </div>
    `;
  }).join('');
}

function providerBadge(provider) {
  const map = {
    aws: '<span class="provider-badge provider-badge--aws">AWS</span>',
    gcp: '<span class="provider-badge provider-badge--gcp">GCP</span>',
    azure: '<span class="provider-badge provider-badge--azure">Azure</span>',
    huggingface: '<span class="provider-badge provider-badge--hf">🤗 HF</span>',
    general: '<span class="provider-badge provider-badge--general">General</span>'
  };
  return map[provider] || provider;
}

function providerLogo(provider) {
  const map = {
    aws: '<span class="provider-logo provider-logo--aws">AWS</span>',
    gcp: '<span class="provider-logo provider-logo--gcp">GCP</span>',
    azure: '<span class="provider-logo provider-logo--azure">Azure</span>',
    huggingface: '<span class="provider-logo provider-logo--hf">🤗</span>'
  };
  return map[provider] || provider;
}

function toggleProjectComplete(id) {
  const completed = store.get('completed.projects');
  if (completed.includes(id)) {
    showToast('Already completed! 🎉', 'info');
    return;
  }
  markItemComplete('projects', id);
  renderProjects();
  renderTimeline();
  renderBadges();
}

function startCert(id) {
  markCertInProgress(id);
  renderCertifications();
}

function passCert(id) {
  markItemComplete('certifications', id);
  renderCertifications();
}

function completeCourse(id) {
  if (store.get('completed.courses').includes(id)) return;
  markItemComplete('courses', id);
  renderCourses();
}

function completeChannel(id) {
  if (store.get('completed.channels').includes(id)) return;
  markItemComplete('channels', id);
  renderChannels();
}

function renderSection(type) {
  const map = { projects: renderProjects, certifications: renderCertifications, courses: renderCourses, channels: renderChannels };
  if (map[type]) map[type]();
}

function initModals() {
  const modal = document.getElementById('project-modal');
  if (!modal) return;

  modal.addEventListener('click', e => {
    if (e.target === modal) closeModal();
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
  });
}

function openProjectModal(id) {
  const project = DATA.projects.find(p => p.id === id);
  if (!project) return;

  const modal = document.getElementById('project-modal');
  const title = document.getElementById('modal-title');
  const body = document.getElementById('modal-body');
  if (!modal || !title || !body) return;

  title.textContent = `${project.num !== '00' ? 'P' + project.num + ': ' : ''}${project.title}`;

  const toolsList = project.tools.map(t => `<code class="tool-chip">${t}</code>`).join(' ');
  const outcomesList = project.outcomes.map(o => `<li>${o}</li>`).join('');

  body.innerHTML = `
    <div class="modal-meta-row">
      <div class="modal-meta-item"><span class="modal-meta-label">Timeline</span><span>${project.week}</span></div>
      <div class="modal-meta-item"><span class="modal-meta-label">XP Reward</span><span class="xp-reward">+${project.xp} XP</span></div>
      <div class="modal-meta-item"><span class="modal-meta-label">Est. Cost</span><span>${project.estimatedCost}</span></div>
    </div>
    <div class="modal-section">
      <h4>🛠 Tools & Technologies</h4>
      <div class="tools-row">${toolsList}</div>
    </div>
    <div class="modal-section">
      <h4>🎯 Expected Outcomes</h4>
      <ul class="outcomes-list">${outcomesList}</ul>
    </div>
    <div class="modal-section" id="modal-requirements">
      <h4>📋 Full Requirements</h4>
      <div id="modal-req-content" style="color:var(--text-muted);font-size:.85rem">Loading requirements…</div>
    </div>
  `;

  modal.classList.add('modal--open');
  document.body.style.overflow = 'hidden';

  if (project.reqPath) {
    fetch(project.reqPath)
      .then(r => r.ok ? r.text() : Promise.reject('not found'))
      .then(md => {
        const el = document.getElementById('modal-req-content');
        if (el) el.innerHTML = `<div class="requirements-content">${markdownToHTML(md)}</div>`;
      })
      .catch(() => {
        const el = document.getElementById('modal-req-content');
        if (el) el.innerHTML = `<p style="color:var(--text-muted)">Requirements file not yet created — see <code>${project.reqPath}</code> in the project folder.</p>`;
      });
  }
}

function closeModal() {
  const modal = document.getElementById('project-modal');
  if (modal) {
    modal.classList.remove('modal--open');
    document.body.style.overflow = '';
  }
}

function markdownToHTML(md) {
  if (!md) return '';
  return md
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang}">${escapeHTML(code.trim())}</code></pre>`)
    .replace(/^#### (.+)$/gm, '<h5>$1</h5>')
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^\- (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    .replace(/(<li>[\s\S]*?<\/li>)+/g, m => `<ul>${m}</ul>`)
    .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(?!<[hup])(.+)$/gm, (m) => m.startsWith('<') ? m : `<p>${m}</p>`);
}

function escapeHTML(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function initThemeToggle() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store.set('user.theme', next);
    updateThemeIcon(next);
  });
}

function updateThemeIcon(theme) {
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function showWelcomeModal(syncEnabled = false) {
  const existing = document.getElementById('welcome-modal');
  if (existing) existing.remove();

  const githubOption = syncEnabled ? `
    <div class="welcome-divider">
      <span>or sign in to sync across devices</span>
    </div>
    <button class="btn-github-signin" onclick="welcomeGitHubSignIn()" style="width:100%;padding:.75rem;display:flex;align-items:center;justify-content:center;gap:.6rem;background:#24292f;color:#fff;border:none;border-radius:var(--radius-sm);font-size:1rem;cursor:pointer;margin-top:.25rem;font-family:var(--font)">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="white"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
      Continue with GitHub
    </button>
    <p style="font-size:.75rem;color:var(--text-muted);margin-top:.5rem">Progress syncs across all your devices automatically</p>
  ` : '';

  const overlay = document.createElement('div');
  overlay.id = 'welcome-modal';
  overlay.className = 'modal-overlay modal--open';
  overlay.style.zIndex = '600';
  overlay.innerHTML = `
    <div class="modal-box" style="max-width:460px;text-align:center;padding:2.5rem 2rem">
      <div style="font-size:3rem;margin-bottom:0.75rem">🧠</div>
      <h2 style="margin-bottom:0.5rem">Welcome to AI Architect Journey</h2>
      <p style="margin-bottom:1.5rem;color:var(--text-secondary)">
        6 months · 12 projects · Multi-cloud certifications<br>
        What should we call you?
      </p>
      <input id="welcome-name-input" type="text" placeholder="Your name…"
        style="width:100%;padding:.75rem 1rem;border-radius:var(--radius-sm);border:1px solid var(--border);
               background:var(--bg-tertiary);color:var(--text-primary);font-size:1rem;margin-bottom:1rem;
               outline:none;font-family:var(--font)"
        maxlength="30">
      <button class="btn btn--primary" style="width:100%;padding:.75rem" onclick="submitWelcomeName()">
        🚀 Begin My Journey
      </button>
      ${githubOption}
    </div>
  `;
  document.body.appendChild(overlay);

  const input = overlay.querySelector('#welcome-name-input');
  if (input) {
    input.focus();
    input.addEventListener('keydown', e => { if (e.key === 'Enter') submitWelcomeName(); });
  }
}

async function welcomeGitHubSignIn() {
  const btn = document.querySelector('.btn-github-signin');
  if (btn) { btn.disabled = true; btn.textContent = 'Redirecting to GitHub…'; }
  try {
    await supabaseSync.signInWithGitHub();
    // Page redirects — execution stops here
  } catch (err) {
    showToast('Sign-in failed: ' + err.message, 'error');
    if (btn) { btn.disabled = false; btn.textContent = 'Continue with GitHub'; }
  }
}

function submitWelcomeName() {
  const input = document.getElementById('welcome-name-input');
  const name = (input ? input.value.trim() : '') || 'Architect';
  store.set('user.name', name);
  const overlay = document.getElementById('welcome-modal');
  if (overlay) overlay.remove();
  updateHeaderStats();
  showToast(`Welcome, ${name}! Your journey begins. 🚀`, 'info');
}

function initProfileEdit() {
  const btn = document.getElementById('edit-profile');
  if (!btn) return;
  btn.addEventListener('click', () => showNameEditModal());
}

function showNameEditModal() {
  const existing = document.getElementById('name-edit-modal');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'name-edit-modal';
  overlay.className = 'modal-overlay modal--open';
  overlay.style.zIndex = '600';
  const current = store.get('user.name') || '';
  overlay.innerHTML = `
    <div class="modal-box" style="max-width:380px;padding:2rem">
      <div class="modal-header" style="border:none;padding:0 0 1rem">
        <h2 class="modal-title">Update Your Name</h2>
        <button class="modal-close" onclick="document.getElementById('name-edit-modal').remove()">✕</button>
      </div>
      <input id="name-edit-input" type="text" value="${current}" placeholder="Your name…"
        style="width:100%;padding:.75rem 1rem;border-radius:var(--radius-sm);border:1px solid var(--border);
               background:var(--bg-tertiary);color:var(--text-primary);font-size:1rem;margin-bottom:1rem;
               outline:none;font-family:var(--font)"
        maxlength="30">
      <button class="btn btn--primary" style="width:100%" onclick="saveEditedName()">Save</button>
    </div>
  `;
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
  const input = overlay.querySelector('#name-edit-input');
  if (input) { input.focus(); input.select(); input.addEventListener('keydown', e => { if (e.key === 'Enter') saveEditedName(); }); }
}

function saveEditedName() {
  const input = document.getElementById('name-edit-input');
  const name = input ? input.value.trim() : '';
  if (name) {
    store.set('user.name', name);
    updateHeaderStats();
    showToast(`Name updated to ${name}!`, 'info');
  }
  const overlay = document.getElementById('name-edit-modal');
  if (overlay) overlay.remove();
}

function initExportImport() {
  const exportBtn = document.getElementById('export-progress');
  const importBtn = document.getElementById('import-progress');
  const resetBtn = document.getElementById('reset-progress');

  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      const data = store.export();
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-journey-progress-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Progress exported! 📥', 'info');
    });
  }

  if (importBtn) {
    importBtn.addEventListener('click', () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json';
      input.onchange = e => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = ev => {
          if (store.import(ev.target.result)) {
            location.reload();
          } else {
            showToast('Invalid progress file', 'error');
          }
        };
        reader.readAsText(file);
      };
      input.click();
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (confirm('Reset ALL progress? This cannot be undone.')) {
        store.reset();
        location.reload();
      }
    });
  }
}
