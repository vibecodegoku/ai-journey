const STORAGE_KEY = 'ai_architect_journey_v1';

const DEFAULT_STATE = {
  user: {
    name: '',
    xp: 0,
    level: 'Apprentice',
    streak: 0,
    lastLogin: null,
    streakSaverUsed: false,
    theme: 'dark',
    joinedDate: null
  },
  completed: {
    projects: [],
    certifications: [],
    courses: [],
    channels: []
  },
  inProgress: {
    certifications: []
  },
  badges: {
    earned: []
  },
  progress: {
    aws_projects: 0,
    gcp_projects: 0,
    azure_projects: 0,
    hf_projects: 0,
    paid_certs_passed: 0,
    certs_started: 0,
    tri_cloud_certs: false,
    governance_complete: false,
    category_complete: false
  },
  filters: {
    providers: [],
    cost: [],
    categories: [],
    activeSection: 'all'
  }
};

class Store {
  constructor() {
    this._state = this._load();
    this._listeners = [];
  }

  _load() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return this._merge(DEFAULT_STATE, parsed);
      }
    } catch (e) {}
    return JSON.parse(JSON.stringify(DEFAULT_STATE));
  }

  _merge(defaults, saved) {
    const result = JSON.parse(JSON.stringify(defaults));
    for (const key in saved) {
      if (key in result && typeof result[key] === 'object' && !Array.isArray(result[key])) {
        result[key] = this._merge(result[key], saved[key]);
      } else {
        result[key] = saved[key];
      }
    }
    return result;
  }

  save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this._state));
      // Cloud sync — debounced push runs 2.5 s after last mutation
      if (typeof supabaseSync !== 'undefined' && supabaseSync.isReady()) {
        supabaseSync.schedulePush(this._state);
      }
    } catch (e) {}
  }

  get(path) {
    return path.split('.').reduce((obj, key) => obj && obj[key], this._state);
  }

  set(path, value) {
    const keys = path.split('.');
    const last = keys.pop();
    const target = keys.reduce((obj, key) => obj[key], this._state);
    target[last] = value;
    this.save();
    this._notify(path, value);
  }

  push(path, value) {
    const arr = this.get(path);
    if (Array.isArray(arr) && !arr.includes(value)) {
      arr.push(value);
      this.save();
      this._notify(path, arr);
    }
  }

  subscribe(fn) {
    this._listeners.push(fn);
    return () => { this._listeners = this._listeners.filter(l => l !== fn); };
  }

  _notify(path, value) {
    this._listeners.forEach(fn => fn(path, value));
  }

  reset() {
    localStorage.removeItem(STORAGE_KEY);
    this._state = JSON.parse(JSON.stringify(DEFAULT_STATE));
    this.save();
    this._notify('reset', null);
  }

  export() {
    return JSON.stringify(this._state, null, 2);
  }

  import(json) {
    try {
      const data = JSON.parse(json);
      this._state = this._merge(DEFAULT_STATE, data);
      this.save();
      this._notify('import', null);
      return true;
    } catch (e) {
      return false;
    }
  }
}

const store = new Store();
