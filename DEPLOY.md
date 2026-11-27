# Deploying GymRec (website + model) with Neon (database)

Two ways to host the same Docker image; the database is Neon either way:

* **[Google Cloud Run](#google-cloud-run)**: free monthly allowance, sleeps when unused. Use this
  if Hugging Face reports a CPU-basic quota limit for your account.
* **[Hugging Face Spaces](#hugging-face-spaces)**: free CPU tier, when your account gets it.

# Hugging Face Spaces

The website runs as one Docker container on a free Hugging Face **Docker Space**; the data
lives in **Neon** Postgres (already set up with `neon deploy`, see `.env`). The trained model
(`ml_models/gymrec_model_two_stage-20261125-1933/`) is part of the repository, so it ships with
the code.

```
browser (camera) ──HTTPS──► Space: gunicorn + Django + YOLO-pose + two-stage classifier
                                     │ pooled connection (DATABASE_URL)
                                     ▼
                               Neon Postgres (users, workouts, model registry)
```

## 0. Before the first deploy (on the laptop, both hosts)

The Neon database must have the tables and the model registered. Already done for the current
Neon branch; repeat after switching to a new Neon project or branch:

```bash
source .venv/bin/activate
python manage.py migrate                                                  # uses DATABASE_URL_UNPOOLED
python manage.py install_model ml_models/gymrec_model_two_stage-20261125-1933
python manage.py install_form_profiles ~/Downloads/form_profiles.json     # optional, once you have it
```

## 1. Create the Space

huggingface.co → **New** → **Space**:
* Space name: e.g. `gymrec`
* SDK: **Docker** → *Blank*
* Hardware: **CPU basic** (free)
* Visibility: your choice (a private Space can only be opened by you while signed in)

## 2. Add the secrets

Space → **Settings** → **Variables and secrets** → **New secret** for each:

| Name | Value |
|---|---|
| `DATABASE_URL` | from `.env` (the `-pooler` one) |
| `DATABASE_URL_UNPOOLED` | from `.env` (used by `migrate` at start-up) |
| `DJANGO_SECRET_KEY` | a new random value: `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
| `DJANGO_ALLOWED_HOSTS` | `.hf.space` |

Already set in the `Dockerfile` (override as variables if needed): `DJANGO_DEBUG=false`,
`RECOGNITION_ALLOW_SIMULATION=false`, `RECOGNITION_INPUT_SIZE=640`,
`DJANGO_CSRF_TRUSTED_ORIGINS=https://*.hf.space` (settings default).

## 3. Push the code

Hugging Face only accepts model files (`*.pt`, `*.joblib`) through Git LFS, while this repository
keeps them as plain git files. `deploy/push_to_space.sh` pushes a one-commit snapshot of your
current commit with the model files in LFS, replacing what the Space had. Your branches and
GitHub history are not changed.

One-time setup: install Git LFS (`brew install git-lfs`), create a **write** access token
(huggingface.co → Settings → Access Tokens), and add the Space as a remote:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/gymrec
```

Then, for the first deploy and every update (commit first):

```bash
deploy/push_to_space.sh
```

When git asks for a password, paste the access token. The Space builds the image (about
10–15 minutes the first time: PyTorch, ultralytics) and starts it. Build and run logs are in the
Space's **Logs** tab. `DRY_RUN=1 deploy/push_to_space.sh` shows what would be pushed.

If the Space was created with another SDK (e.g. Static), the `sdk: docker` line at the top of
`README.md` switches it; if it doesn't, change it under the Space's **Settings**.

## 4. Open it

`https://<your-username>-gymrec.hf.space`, or the Space page's ⋮ menu → **Embed this Space**
→ *Direct URL*. Use that direct address for workouts: browsers only allow the camera on HTTPS,
and the direct page avoids iframe camera/cookie restrictions on the huggingface.co page.

The Neon database starts empty: sign up on the site. For demo data, run `python manage.py
seed_demo` on the laptop (it writes to Neon through `.env`).

## Things to know

* **Sleep:** a free Space sleeps when unused for a while; the next visit wakes it (a minute or two).
  Neon's compute also scales to zero and wakes on the first query (well under a second).
* **Speed:** CPU basic is slower than a laptop; the live page simply processes fewer frames per
  second. The rep counter is built for 3+ frames per second.
* **New model:** install it locally (`install_model`, which also registers it in Neon), add its
  folder to the `!ml_models/...` exceptions in `.gitignore` (and remove the old one), commit, then
  run `deploy/push_to_space.sh`.
* **Never** commit `.env`: the secrets above are the only place the database password belongs.

# Google Cloud Run

Cloud Build builds the `Dockerfile` in Google's network (only the source, about 30 MB, is uploaded;
`.gcloudignore` keeps `.env` and other secrets out), and Cloud Run runs it with HTTPS at
`https://gymrec-<id>.<region>.run.app`.

## One-time setup

1. **Google Cloud account and project:** console.cloud.google.com → create a project (e.g.
   `gymrec`) → **Billing**: link a billing account. Cloud Run needs one even when usage stays inside
   the free monthly allowance (new accounts also get trial credit).
2. **Google Cloud CLI** on the laptop, then sign in and select the project:
   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth login
   gcloud config set project <your-project-id>
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
3. **Settings:** `deploy/cloudrun.env.yaml` holds the Neon URLs, a secret key and the allowed hosts
   (template: `deploy/cloudrun.env.example.yaml`). It is git-ignored and not uploaded with the source;
   `gcloud` sends its values to Cloud Run as environment variables.

## Deploy and update

```bash
deploy/deploy_cloud_run.sh
```

The first run asks to create an Artifact Registry repository (answer **Y**) and takes about
10–15 minutes; later deploys reuse cached layers. It prints the live URL at the end. Logs:
Cloud Run console → service `gymrec` → **Logs**.

Settings used: region `us-central1` (close to the Neon database in us-east-2), 2 vCPU / 2 GiB
(the models peak at about 0.6 GB), scale to zero when idle, at most 2 instances, CPU boost at
start-up. Change them in `deploy/deploy_cloud_run.sh`.

## Costs and limits

* Request-based billing: you pay only while requests are processed, and the free allowance
  per billing account per month (180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests at
  the time of writing) is about 25 hours of active processing with these settings. Every camera
  frame is one request. Check https://cloud.google.com/run/pricing and set a budget alert under
  **Billing → Budgets & alerts**.
* After an idle period the first visit starts an instance (about 10–20 s: loading PyTorch, YOLO
  and the models).
