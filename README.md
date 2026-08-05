# Unemployment Stack

A local kanban board for a job search that fixes one of the most annoying job board problems: **having to actually enter and move your jobs**. Drag a card between **Applied →
Interviewing → Offer → Rejected** and it stamps the date it got there and how
many days that took. Unemploymeny Stack reads your Gmail for confirmations and
rejections, scrapes open postings off the web, and ranks them against your
resume — all of it on your machine, with a local LLM. Nothing is sent anywhere. 
There are also views to understand statistics around timing of interviews/ rejections, a github style activity tracker for job applications, a sankey diagram, and salary/skills research based on job title.  

## Features

| Tab                 | What it does                                                                       |
|---------------------|------------------------------------------------------------------------------------|
| **Board**           | Add / drag / edit job cards. Milestone dates and day-counts are automatic.          |
| **Timing, Activity, Flow, Salary** | Charts over your board and the postings you've scraped.              |
| **Sync** (header)   | Reads Gmail, classifies each mail with a local LLM, proposes cards.                 |
| **Job search**      | **Scan** pulls open postings from two public feeds; **Check** verifies each is still live and extracts salary / years / skills. |
| **Resume analysis** | Upload a PDF, get every live posting scored against it on five features.            |

Only the Board is required. Every other tab is optional and degrades to "not
installed" without breaking the app.

## Requirements

| For                             | Needs                                                        |
|---------------------------------|--------------------------------------------------------------|
| The board                       | Python 3.8+, stdlib only                                     |
| Job search, Sync, Resume        | `pip install -r requirements.txt` (`tqdm`, `numpy`, `sentence-transformers`, `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`) |
| Sync, Check postings            | [Ollama](https://ollama.com) with a model pulled (default `qwen3.6:35b`) |
| Sync                            | A Google OAuth client + token — [below](#gmail-access-one-time) |
| Resume PDF upload               | `poppler-utils` (for `pdftotext`) — or paste the text instead |

`sentence-transformers` depends on PyTorch. It is recommended to install PyTorch with your desired GPU backend before installing
the rest of the dependcies. This will make the embedding process much faster.   

## Install

```bash
git clone <this repo> && cd unemployment_stack
python3 app.py                            # board only — done, nothing installed
```

For the optional tabs:

```bash
pip install --user -r requirements.txt
ollama pull qwen3.6:35b                   # Sync + Check postings
sudo dnf install poppler-utils            # or apt install poppler-utils / or install it with anaconda on windows
python3 embed.py                          # Resume analysis: embed every posting (~45s on GPU)
```

## Run

```bash
python3 app.py
```

Serves `http://127.0.0.1:8000` and opens your browser; `Ctrl+C` stops it.
`./run-tracker.sh` does the same from any directory.


## Gmail access (one-time)

Only needed for **Sync**. Two files end up in this folder, and neither exists
until you make it:

| File               | What it is                                     | Where it comes from             |
|--------------------|------------------------------------------------|---------------------------------|
| `credentials.json` | the OAuth **client** — identifies this app     | you download it (steps 1–4)     |
| `token.json`       | your **token** — grants read-only Gmail access | written by the sign-in (step 5) |

These are secrets: `.gitignore` already excludes them, do not change this (unless you want other people to read your emails I guess)

**1. Create a Google Cloud project.** [Cloud
Console](https://console.cloud.google.com) → project dropdown → **New Project**
→ name it (e.g. `unemployment-stack`) → **Create**, and select it.

**2. Enable the Gmail API.** **APIs & Services → Library** → search **Gmail
API** → **Enable**.

**3. Configure the consent screen.** **APIs & Services → OAuth consent screen**
(newer consoles: **Google Auth Platform → Branding / Audience**): user type
**External**; fill in app name and your email; skip **Scopes** (the script asks
for `gmail.readonly` at sign-in); add your own address under **Test users**;
leave publishing status on **Testing**.

**4. Create the OAuth client → `credentials.json`.** **APIs & Services →
Credentials → Create credentials → OAuth client ID** → application type
**Desktop app** → **Create** → **Download JSON**. Rename the download to
exactly `credentials.json` and put it next to `app.py`.

**5. Sign in once → `token.json`.**

```bash
python3 ingest.py
```

A browser tab opens on a random `127.0.0.1` port (the Desktop client type
allows that, so there's no redirect URI to configure). Pick your account, grant
**read-only** Gmail. The app is unverified, so you'll hit **"Google hasn't
verified this app"** → **Advanced → Go to unemployment-stack (unsafe) → Continue** —
it's your own client reading your own mail. On success `token.json` is written
here and the scan starts.

### Using the token

Nothing to pass or export; everything reads `token.json` from this folder.

- `python3 ingest.py` and the header's **Sync** button both use it.
- Expired *access* tokens refresh silently, no prompt.
- When the *refresh* token dies, the header's **Reconnect Gmail** button
  reopens the consent tab. `python3 ingest.py` does the same from the shell.
- Starting over — revoked access, different account — is `rm token.json`, then
  `python3 ingest.py`.
- **In Testing, Google expires the refresh token after ~7 days**, so the
  sign-in repeats about weekly. Set the OAuth app to **In production** to stop
  that; the unverified-app screen stays, the expiry doesn't.

Sign-in failing? See [Gmail troubleshooting](DOCS.md#gmail-troubleshooting).

## Data & privacy

`tracker.db` (board) and `research.db` (postings, resume) are SQLite files next
to `app.py`, created on first run — back them up or delete them like any file. Email
bodies, resume text and postings go only to your local Ollama; the sole
outbound traffic is Gmail's API, the two public job feeds, and fetching posting
pages to check they're alive.

## More


This app was originally designed with qwen3.6:35b in mind but you can choose your own LLM to fit your system's requirements. 
I also tested this using qwen3.5:4b and it runs much faster but makes more mistakes. 
The intial ingest and job title enrichment can take a pretty long time (hours) when using larger local models even with heavy parallelism. 
If you have a GPU it is recommended to increase the batch size on the ebmeddings to take advantage.     


[DOCS.md](DOCS.md) — HTTP API, CLI flags for every script, how postings are
scored and how liveness is decided, and the tests.
