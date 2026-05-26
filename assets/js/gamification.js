const LEVELS = [
  { name: 'Apprentice', minXP: 0, emoji: '🌱', color: '#94a3b8' },
  { name: 'Practitioner', minXP: 1000, emoji: '⚡', color: '#60a5fa' },
  { name: 'Associate', minXP: 3000, emoji: '🔧', color: '#34d399' },
  { name: 'Specialist', minXP: 6000, emoji: '🎯', color: '#a78bfa' },
  { name: 'Professional', minXP: 10000, emoji: '💡', color: '#f59e0b' },
  { name: 'Expert', minXP: 15000, emoji: '🚀', color: '#f97316' },
  { name: 'Architect', minXP: 20000, emoji: '👑', color: '#ec4899' }
];

function getLevelForXP(xp) {
  for (let i = LEVELS.length - 1; i >= 0; i--) {
    if (xp >= LEVELS[i].minXP) return LEVELS[i];
  }
  return LEVELS[0];
}

function getNextLevel(xp) {
  for (let i = 0; i < LEVELS.length; i++) {
    if (xp < LEVELS[i].minXP) return LEVELS[i];
  }
  return null;
}

function getLevelProgress(xp) {
  const current = getLevelForXP(xp);
  const next = getNextLevel(xp);
  if (!next) return { pct: 100, xpInLevel: xp - current.minXP, xpNeeded: 0 };
  const xpInLevel = xp - current.minXP;
  const xpNeeded = next.minXP - current.minXP;
  return { pct: Math.round((xpInLevel / xpNeeded) * 100), xpInLevel, xpNeeded, nextName: next.name };
}

function awardXP(amount, reason) {
  const oldXP = store.get('user.xp');
  const newXP = oldXP + amount;
  const oldLevel = getLevelForXP(oldXP);
  const newLevel = getLevelForXP(newXP);

  store.set('user.xp', newXP);
  store.set('user.level', newLevel.name);

  showToast(`+${amount} XP — ${reason}`, 'xp');

  if (oldLevel.name !== newLevel.name) {
    setTimeout(() => showLevelUp(newLevel), 800);
  }

  updateHeaderStats();
  checkBadges();
}

function checkBadges() {
  const completed = store.get('completed');
  const progress = store.get('progress');
  const earned = store.get('badges.earned');
  const xp = store.get('user.xp');
  const streak = store.get('user.streak');

  const conditions = {
    'first-steps': completed.courses.length + completed.channels.length >= 1,
    'builder': completed.projects.length >= 1,
    'cloud-curious': store.get('inProgress.certifications').length + completed.certifications.length >= 1,
    'aws-pioneer': progress.aws_projects >= 1,
    'gcp-explorer': progress.gcp_projects >= 1,
    'azure-navigator': progress.azure_projects >= 1,
    'prompt-wizard': completed.projects.includes('p05'),
    'rag-master': completed.projects.includes('p06'),
    'agent-whisperer': completed.projects.includes('p08'),
    'orchestrator': completed.projects.includes('p09'),
    'mlops-hero': completed.projects.includes('p10'),
    'multi-cloud': completed.projects.includes('p11'),
    'project-champion': completed.projects.length >= 5,
    'halfway': completed.projects.length >= 6,
    'completionist': completed.projects.length >= 12,
    'certified': progress.paid_certs_passed >= 1,
    'tri-cloud': progress.tri_cloud_certs,
    'streak-7': streak >= 7,
    'streak-30': streak >= 30,
    'open-source': completed.projects.includes('p05'),
    'deep-diver': progress.category_complete,
    'architect-elite': xp >= 20000,
    'governance': progress.governance_complete
  };

  for (const [badgeId, conditionMet] of Object.entries(conditions)) {
    if (conditionMet && !earned.includes(badgeId)) {
      store.push('badges.earned', badgeId);
      const badge = DATA.badges.find(b => b.id === badgeId);
      if (badge) {
        setTimeout(() => {
          showToast(`${badge.emoji} Badge Unlocked: ${badge.name}!`, 'badge');
          showConfetti();
        }, 400);
      }
    }
  }

  renderBadges();
}

function updateStreak() {
  const today = new Date().toDateString();
  const lastLogin = store.get('user.lastLogin');

  if (!lastLogin) {
    store.set('user.lastLogin', today);
    store.set('user.streak', 1);
    return;
  }

  if (lastLogin === today) return;

  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toDateString();

  if (lastLogin === yesterdayStr) {
    const newStreak = store.get('user.streak') + 1;
    store.set('user.streak', newStreak);
    if (newStreak % 7 === 0) showToast(`🔥 ${newStreak}-day streak! Keep going!`, 'streak');
  } else {
    const streakSaverUsed = store.get('user.streakSaverUsed');
    const twoDaysAgo = new Date();
    twoDaysAgo.setDate(twoDaysAgo.getDate() - 2);
    if (!streakSaverUsed && lastLogin === twoDaysAgo.toDateString()) {
      store.set('user.streakSaverUsed', true);
      showToast('🛡️ Streak Saver used! Streak preserved.', 'streak');
    } else {
      store.set('user.streak', 1);
      store.set('user.streakSaverUsed', false);
    }
  }

  store.set('user.lastLogin', today);
  checkBadges();
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = message;
  container.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add('toast--visible'));

  setTimeout(() => {
    toast.classList.remove('toast--visible');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function showLevelUp(level) {
  const overlay = document.createElement('div');
  overlay.className = 'level-up-overlay';
  overlay.innerHTML = `
    <div class="level-up-card">
      <div class="level-up-emoji">${level.emoji}</div>
      <div class="level-up-title">Level Up!</div>
      <div class="level-up-name">${level.name}</div>
      <div class="level-up-sub">You're now an AI ${level.name}</div>
      <button class="btn btn--primary" onclick="this.closest('.level-up-overlay').remove()">Awesome! 🎉</button>
    </div>
  `;
  document.body.appendChild(overlay);
  showConfetti();
}

function showConfetti() {
  const canvas = document.createElement('canvas');
  canvas.className = 'confetti-canvas';
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  const particles = Array.from({ length: 80 }, () => ({
    x: Math.random() * canvas.width,
    y: -10,
    vx: (Math.random() - 0.5) * 4,
    vy: Math.random() * 3 + 2,
    color: ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ec4899'][Math.floor(Math.random() * 5)],
    size: Math.random() * 8 + 4,
    rotation: Math.random() * 360
  }));

  let frame = 0;
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.05;
      p.rotation += 3;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate((p.rotation * Math.PI) / 180);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
      ctx.restore();
    });
    frame++;
    if (frame < 120) requestAnimationFrame(animate);
    else canvas.remove();
  }
  animate();
}

function updateHeaderStats() {
  const xp = store.get('user.xp');
  const level = getLevelForXP(xp);
  const streak = store.get('user.streak');
  const progress = getLevelProgress(xp);
  const name = store.get('user.name');

  const elXP = document.getElementById('stat-xp');
  const elLevel = document.getElementById('stat-level');
  const elStreak = document.getElementById('stat-streak');
  const elName = document.getElementById('user-name-display');
  const elLevelBar = document.getElementById('level-progress-bar');
  const elLevelText = document.getElementById('level-progress-text');

  if (elXP) elXP.textContent = xp.toLocaleString();
  if (elLevel) { elLevel.textContent = `${level.emoji} ${level.name}`; elLevel.style.color = level.color; }
  if (elStreak) elStreak.textContent = `${streak}${streak >= 7 ? ' 🔥' : ''}`;
  if (elName && name) elName.textContent = name;
  if (elLevelBar) elLevelBar.style.width = `${progress.pct}%`;
  if (elLevelText) {
    const next = getNextLevel(xp);
    elLevelText.textContent = next
      ? `${progress.xpInLevel.toLocaleString()} / ${progress.xpNeeded.toLocaleString()} XP to ${next.name}`
      : 'Maximum Level Reached 👑';
  }

  updateHeroDashboard();
}

function updateHeroDashboard() {
  const completed = store.get('completed');
  const xp = store.get('user.xp');
  const totalItems = DATA.projects.length + DATA.certifications.filter(c => c.cost === 'paid').length + DATA.courses.length;
  const completedItems = completed.projects.length + completed.certifications.length + completed.courses.length;
  const overallPct = Math.round((completedItems / totalItems) * 100);

  const elProjects = document.getElementById('hero-projects');
  const elCerts = document.getElementById('hero-certs');
  const elXP = document.getElementById('hero-xp');
  const elProgress = document.getElementById('hero-progress-pct');
  const ring = document.getElementById('progress-ring-fill');

  if (elProjects) elProjects.textContent = completed.projects.length;
  if (elCerts) elCerts.textContent = completed.certifications.length;
  if (elXP) elXP.textContent = xp.toLocaleString();
  if (elProgress) elProgress.textContent = `${overallPct}%`;
  if (ring) {
    const circumference = 2 * Math.PI * 54;
    ring.style.strokeDasharray = circumference;
    ring.style.strokeDashoffset = circumference - (circumference * overallPct) / 100;
  }

  updateNextMilestone();
}

function updateNextMilestone() {
  const el = document.getElementById('next-milestone');
  if (!el) return;

  const xp = store.get('user.xp');
  const next = getNextLevel(xp);
  const completed = store.get('completed');
  const projectsLeft = 5 - completed.projects.length;

  if (!next) {
    el.textContent = '👑 Maximum level reached — you\'re an Architect!';
    return;
  }

  const xpLeft = next.minXP - xp;
  el.textContent = `${xpLeft.toLocaleString()} XP to reach ${next.name}${projectsLeft > 0 ? ` — complete ${Math.max(0, projectsLeft)} more project${projectsLeft !== 1 ? 's' : ''} to hit Project Champion 🏆` : ''}`;
}

function markItemComplete(type, id) {
  const completed = store.get(`completed.${type}`);
  if (completed.includes(id)) {
    showToast('Already completed!', 'info');
    return;
  }

  store.push(`completed.${type}`, id);

  let xpAmount = 0;
  let reason = '';

  if (type === 'projects') {
    const project = DATA.projects.find(p => p.id === id);
    if (project) {
      xpAmount = project.xp;
      reason = `Project: ${project.title}`;
      project.providers.forEach(provider => {
        if (['aws', 'gcp', 'azure', 'huggingface'].includes(provider)) {
          const key = `${provider === 'huggingface' ? 'hf' : provider}_projects`;
          store.set(`progress.${key}`, (store.get(`progress.${key}`) || 0) + 1);
        }
      });
    }
    showConfetti();
  } else if (type === 'certifications') {
    const cert = DATA.certifications.find(c => c.id === id);
    if (cert) {
      xpAmount = cert.cost === 'paid' ? 500 : 100;
      reason = `Certification: ${cert.title}`;
      if (cert.cost === 'paid') {
        store.set('progress.paid_certs_passed', store.get('progress.paid_certs_passed') + 1);
        checkTriCloud();
      }
      showConfetti();
    }
  } else if (type === 'courses') {
    const course = DATA.courses.find(c => c.id === id);
    if (course) {
      xpAmount = course.xp || 75;
      reason = `Course: ${course.title}`;
    }
  } else if (type === 'channels') {
    xpAmount = 25;
    reason = 'Video resource viewed';
  }

  awardXP(xpAmount, reason);
  renderSection(type);
}

function checkTriCloud() {
  const completed = store.get('completed.certifications');
  const hasAWS = completed.some(id => id.startsWith('cert-aws-') && DATA.certifications.find(c => c.id === id && c.cost === 'paid'));
  const hasGCP = completed.some(id => id.startsWith('cert-gcp-') && DATA.certifications.find(c => c.id === id && c.cost === 'paid'));
  const hasAzure = completed.some(id => id.startsWith('cert-azure-') && DATA.certifications.find(c => c.id === id && c.cost === 'paid'));
  if (hasAWS && hasGCP && hasAzure) store.set('progress.tri_cloud_certs', true);
}

function markCertInProgress(id) {
  const inProgress = store.get('inProgress.certifications');
  if (!inProgress.includes(id)) {
    store.push('inProgress.certifications', id);
    store.set('progress.certs_started', store.get('progress.certs_started') + 1);
    showToast('📖 Certification marked as in-progress!', 'info');
    checkBadges();
    renderSection('certifications');
  }
}
