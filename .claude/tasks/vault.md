# Local Secrets Vault — AI Craftsman KB

**Recommendation: `~/.ai-craftsman-kb/.env` + `direnv`**

Here's why this is the right fit for a local-first solo tool, and how to
set it up.

---

## The Options (ranked for this use case)

| Option | Security | Friction | Best for |
|--------|----------|----------|---------|
| **`~/.ai-craftsman-kb/.env` + direnv** | Good | Low | ✅ This project |
| macOS Keychain + shell helper | Excellent | Medium | High-security needs |
| 1Password CLI (`op run`) | Excellent | Medium-high | Already a 1P user |
| Shell profile (`~/.zshrc`) | Poor | None | Never — leaks to all processes |
| `.env` in repo root | Very poor | None | Never — risk of commit |

---

## Recommended Setup: direnv + `~/.ai-craftsman-kb/.env`

### Why direnv?

- Automatically loads/unloads env vars when you `cd` into the project
- `.envrc` just sources your external `.env` file — secrets live outside git
- Standard tool in the macOS/dev ecosystem, works with zsh/bash
- Zero per-command friction once set up

### Install

```bash
brew install direnv

# Add to ~/.zshrc (or ~/.bashrc):
eval "$(direnv hook zsh)"

# Reload shell
source ~/.zshrc
```

### Create your secrets file

```bash
mkdir -p ~/.ai-craftsman-kb

cat > ~/.ai-craftsman-kb/.env << 'EOF'
# AI Craftsman KB — local secrets
# This file lives outside the git repo. Never commit it.

export OPENAI_API_KEY="sk-proj-..."
export OPENROUTER_API_KEY="sk-or-v1-..."
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export YOUTUBE_API_KEY="AIza..."
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
EOF

# Lock it down — readable only by you
chmod 600 ~/.ai-craftsman-kb/.env
```

### Create the project `.envrc`

In the project root (`ai-craftsman-kb/`):

```bash
cat > .envrc << 'EOF'
# Load secrets from outside the repo
dotenv ~/.ai-craftsman-kb/.env
EOF

# Approve it (direnv requires explicit approval for each .envrc)
direnv allow
```

The `.envrc` file CAN be committed (it contains no secrets, just a path).
It's already in `.gitignore` by default — confirm with `git check-ignore .envrc`.

### Verify

```bash
cd ai-craftsman-kb
# Should see: direnv: loading ~/.ai-craftsman-kb/.env
echo $OPENAI_API_KEY   # should print your key
uv run cr doctor       # should show ✓ for configured keys
```

---

## Alternative: macOS Keychain

If you don't want a plaintext `.env` file (even chmod 600), use Keychain:

```bash
# Store a secret
security add-generic-password \
  -a "$USER" \
  -s "OPENAI_API_KEY" \
  -w "sk-proj-..."

# Read it in your .envrc or shell profile
export OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w)
```

Downsides: slightly more setup, and each new terminal prompts macOS for
keychain access (can be set to "always allow" for the terminal app).

---

## Alternative: 1Password CLI

If you use 1Password, `op run` is excellent:

```bash
# Store secrets in 1Password vault, then:
op run --env-file=.env.1p -- uv run cr ingest

# Or in .envrc:
export OPENAI_API_KEY=$(op read "op://Private/OpenAI/api_key")
```

---

## What NOT to Do

- **Don't** put secrets in `~/.zshrc` or `~/.zprofile` — they get exported to
  every process on your system, including ones that shouldn't see them.
- **Don't** put a `.env` file in the project root unless it's in `.gitignore`
  AND you've verified that with `git check-ignore .env`.
- **Don't** hardcode keys in `settings.yaml` and commit it — the file uses
  `${VAR}` interpolation specifically to avoid this.
- **Don't** use the same API key for this tool and production systems —
  create dedicated keys with minimal scope.

---

## Rotating Keys

When you need to rotate a key:

```bash
# Update the secrets file
nano ~/.ai-craftsman-kb/.env

# Reload direnv
direnv reload

# Verify
uv run cr doctor
```
