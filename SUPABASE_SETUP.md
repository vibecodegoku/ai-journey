# Enable Cloud Sync — Supabase Setup Guide

Follow these steps once (~10 minutes) to enable cross-device progress sync via Supabase + GitHub OAuth.

## Step 1 — Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) → **New project**
2. Choose a name (e.g. `ai-journey`), set a database password, pick a region close to you
3. Wait ~2 minutes for the project to initialise

## Step 2 — Run the Database Migration

In your Supabase project → **SQL Editor** → paste and run:

```sql
create table public.user_progress (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users not null unique,
  state       jsonb not null default '{}',
  updated_at  timestamptz not null default now()
);

alter table public.user_progress enable row level security;

create policy "own_read"   on public.user_progress
  for select using (auth.uid() = user_id);

create policy "own_insert" on public.user_progress
  for insert with check (auth.uid() = user_id);

create policy "own_update" on public.user_progress
  for update using (auth.uid() = user_id);

create or replace function public.handle_updated_at()
  returns trigger as $$
  begin new.updated_at = now(); return new; end;
  $$ language plpgsql;

create trigger on_progress_updated
  before update on public.user_progress
  for each row execute procedure public.handle_updated_at();
```

## Step 3 — Create a GitHub OAuth App

1. Go to [github.com/settings/applications/new](https://github.com/settings/applications/new)
2. Fill in:
   - **Application name:** AI Architect Journey
   - **Homepage URL:** `https://YOUR_USERNAME.github.io/ai-architect-journey/`
   - **Authorization callback URL:** `https://YOUR_SUPABASE_PROJECT.supabase.co/auth/v1/callback`
     *(find this in Supabase → Auth → Providers → GitHub → Callback URL)*
3. Click **Register application**
4. Copy **Client ID** and generate a **Client Secret**

## Step 4 — Enable GitHub Auth in Supabase

1. Supabase Dashboard → **Authentication → Providers → GitHub**
2. Toggle **Enable**
3. Paste your **Client ID** and **Client Secret**
4. Set **Redirect URL** to your GitHub Pages URL:
   `https://YOUR_USERNAME.github.io/ai-architect-journey/`
5. Click **Save**

## Step 5 — Add Credentials to the App

Open `assets/js/supabase-sync.js` and replace the placeholder values at the top:

```js
// Before:
const SUPABASE_URL      = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';

// After (from Supabase Dashboard → Settings → API):
const SUPABASE_URL      = 'https://abcdefghijkl.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
```

> **The anon key is safe to commit.** Supabase Row Level Security (RLS) ensures each user can only read and write their own data. Never commit the `service_role` key.

## Step 6 — Deploy & Test

```bash
git add assets/js/supabase-sync.js
git commit -m "configure Supabase cloud sync"
git push origin main
```

GitHub Actions will deploy to GitHub Pages. Open the live URL, click **☁️ Sync Progress** in the header, and sign in with GitHub. Your progress will sync automatically on every action.

## How Sync Works

| Situation | Behaviour |
|---|---|
| Not signed in | localStorage only (existing behaviour) |
| Sign in with GitHub | Cloud state loaded; local merged if local has more XP |
| Mark a project complete | Saved locally immediately; pushed to cloud 2.5 s later |
| Open site on another device | After sign in, cloud state is loaded automatically |
| Offline | Falls back to localStorage; resumes sync when back online |
| Sign out | Session cleared; local progress preserved |

## Troubleshooting

| Problem | Fix |
|---|---|
| "☁️ Sync Progress" button not visible | Check browser console — `SUPABASE_URL` may still be the placeholder |
| Redirect loop after GitHub auth | Ensure Redirect URL in Supabase matches your exact GitHub Pages URL |
| "Sync error" indicator | Open DevTools console for details; usually a CORS or RLS issue |
| Progress not loading on second device | Confirm you signed in with the **same** GitHub account |
