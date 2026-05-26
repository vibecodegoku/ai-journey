// ============================================================
//  Supabase Cloud Sync — AI Architect Journey
//  Handles GitHub OAuth + cross-device progress persistence
// ============================================================
//
//  SETUP (one-time, ~10 minutes):
//
//  1. Create a free project at https://supabase.com
//
//  2. In the Supabase SQL editor, run this migration:
//
//     create table public.user_progress (
//       id          uuid primary key default gen_random_uuid(),
//       user_id     uuid references auth.users not null unique,
//       state       jsonb not null default '{}',
//       updated_at  timestamptz not null default now()
//     );
//     alter table public.user_progress enable row level security;
//     create policy "own_read"   on public.user_progress for select using (auth.uid() = user_id);
//     create policy "own_insert" on public.user_progress for insert with check (auth.uid() = user_id);
//     create policy "own_update" on public.user_progress for update using (auth.uid() = user_id);
//     create or replace function public.handle_updated_at()
//       returns trigger as $$ begin new.updated_at = now(); return new; end; $$ language plpgsql;
//     create trigger on_progress_updated before update on public.user_progress
//       for each row execute procedure public.handle_updated_at();
//
//  3. Enable GitHub OAuth:
//     Supabase Dashboard → Authentication → Providers → GitHub
//     Create a GitHub OAuth App at https://github.com/settings/applications/new:
//       - Homepage URL: your GitHub Pages URL (e.g. https://yourname.github.io/ai-architect-journey/)
//       - Callback URL: https://<your-supabase-project>.supabase.co/auth/v1/callback
//     Paste Client ID + Secret into Supabase dashboard.
//
//  4. Copy your project credentials from:
//     Supabase Dashboard → Settings → API
//     Replace the placeholder values below.
//
// ============================================================

const SUPABASE_URL      = 'YOUR_SUPABASE_URL';       // e.g. https://abcdefgh.supabase.co
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';  // safe to commit — RLS protects data

const DB_TABLE          = 'user_progress';
const SYNC_DEBOUNCE_MS  = 2500;   // batch rapid XP/badge changes into one push

// ============================================================

class SupabaseSync {
  constructor() {
    this._client        = null;
    this._user          = null;
    this._pushTimer     = null;
    this._authCallbacks = [];
    this._configured    = false;
    this._ready         = false;
  }

  // Call once on app start. Returns false if not configured.
  init() {
    const isPlaceholder = !SUPABASE_URL || SUPABASE_URL === 'YOUR_SUPABASE_URL';
    if (isPlaceholder || !window.supabase) {
      if (!isPlaceholder) {
        console.warn('[Sync] Supabase JS SDK not loaded.');
      }
      return false;
    }

    try {
      this._client = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: true,   // handles the OAuth redirect
        },
      });
      this._configured = true;

      // Listen for future auth events (sign in / sign out)
      this._client.auth.onAuthStateChange((event, session) => {
        this._user  = session?.user ?? null;
        this._ready = !!this._user;
        this._authCallbacks.forEach(cb => cb(event, this._user));
      });

      // Restore existing session (returns immediately from localStorage)
      this._client.auth.getSession().then(({ data }) => {
        this._user  = data?.session?.user ?? null;
        this._ready = !!this._user;
      });

      return true;
    } catch (err) {
      console.error('[Sync] Init failed:', err);
      return false;
    }
  }

  isConfigured() { return this._configured; }
  isReady()      { return this._ready;      }
  getUser()      { return this._user;       }

  // Register a callback for auth state changes. Returns unsubscribe fn.
  onAuthChange(callback) {
    this._authCallbacks.push(callback);
    return () => { this._authCallbacks = this._authCallbacks.filter(c => c !== callback); };
  }

  // Triggers GitHub OAuth flow — page redirects then returns after auth.
  async signInWithGitHub() {
    if (!this._client) throw new Error('Supabase not initialized');
    const redirectTo = window.location.href.split('?')[0].split('#')[0];
    const { error } = await this._client.auth.signInWithOAuth({
      provider: 'github',
      options: { redirectTo },
    });
    if (error) throw error;
  }

  async signOut() {
    if (!this._client) return;
    await this._client.auth.signOut();
    this._user  = null;
    this._ready = false;
  }

  // Debounced — batches rapid state mutations (XP gain + badge unlock) into one API call.
  schedulePush(state) {
    if (!this.isReady()) return;
    this._setSyncStatus('syncing');
    clearTimeout(this._pushTimer);
    this._pushTimer = setTimeout(() => this._doPush(state), SYNC_DEBOUNCE_MS);
  }

  async _doPush(state) {
    if (!this.isReady()) return;
    try {
      const { error } = await this._client
        .from(DB_TABLE)
        .upsert(
          { user_id: this._user.id, state, updated_at: new Date().toISOString() },
          { onConflict: 'user_id' }
        );
      if (error) throw error;
      this._setSyncStatus('ok');
    } catch (err) {
      console.warn('[Sync] Push failed:', err.message);
      this._setSyncStatus('error');
    }
  }

  // Fetch this user's saved state from Supabase. Returns null if not found.
  async pullState() {
    if (!this.isReady()) return null;
    try {
      const { data, error } = await this._client
        .from(DB_TABLE)
        .select('state, updated_at')
        .eq('user_id', this._user.id)
        .single();

      if (error) {
        if (error.code === 'PGRST116') return null; // row does not exist yet
        throw error;
      }
      return data;
    } catch (err) {
      console.warn('[Sync] Pull failed:', err.message);
      return null;
    }
  }

  // GitHub profile info helpers
  getUserDisplayName() {
    if (!this._user) return null;
    return (
      this._user.user_metadata?.full_name  ||
      this._user.user_metadata?.user_name  ||
      this._user.email?.split('@')[0]      ||
      'GitHub User'
    );
  }

  getUserAvatar() {
    return this._user?.user_metadata?.avatar_url || null;
  }

  // Update the small sync status chip in the header
  _setSyncStatus(status) {
    const el = document.getElementById('sync-indicator');
    if (!el) return;
    const map = {
      syncing: { text: '↑ Syncing…',  cls: 'sync-indicator--syncing' },
      ok:      { text: '☁ Synced',    cls: 'sync-indicator--ok'      },
      error:   { text: '⚠ Sync error', cls: 'sync-indicator--error'   },
    };
    const s = map[status] || {};
    el.textContent = s.text || '';
    el.className   = `sync-indicator ${s.cls || ''}`;

    // Auto-clear "Synced" after 4 s
    if (status === 'ok') {
      clearTimeout(this._clearTimer);
      this._clearTimer = setTimeout(() => {
        el.textContent = '';
        el.className   = 'sync-indicator';
      }, 4000);
    }
  }
}

// Global singleton — referenced by store.js and app.js
const supabaseSync = new SupabaseSync();
