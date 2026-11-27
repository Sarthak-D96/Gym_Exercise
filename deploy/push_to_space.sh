#!/bin/sh
# Push the current commit to the Hugging Face Space (see DEPLOY.md).
#
#   deploy/push_to_space.sh            # remote "space"
#   deploy/push_to_space.sh myremote
#   DRY_RUN=1 deploy/push_to_space.sh  # build the snapshot, check it, don't push
#
# Hugging Face stores binary files (*.pt, *.joblib, ...) only through Git LFS, while this
# repository keeps the model as plain git files (simple for GitHub). So the Space gets a
# single snapshot commit of the current code with the model files in LFS, force-pushed over
# whatever the Space had. Your own branches and the GitHub history are not changed.
set -e
REMOTE=${1:-space}

git lfs version >/dev/null 2>&1 || { echo "Git LFS is missing: brew install git-lfs"; exit 1; }
git remote get-url "$REMOTE" >/dev/null 2>&1 || {
    echo "No git remote '$REMOTE'. Add it: git remote add $REMOTE https://huggingface.co/spaces/<user>/<space>"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "Commit your changes first (git status)."; exit 1; }

SRC=$(git rev-parse --short HEAD)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/gymrec-space.XXXXXX")
cleanup() {
    git worktree remove --force "$TMP" >/dev/null 2>&1 || rm -rf "$TMP"
    git branch -D hf-space-snapshot >/dev/null 2>&1 || true
}
trap cleanup EXIT

git branch -D hf-space-snapshot >/dev/null 2>&1 || true
git worktree add --quiet --detach "$TMP" HEAD
(
    cd "$TMP"
    git checkout --quiet --orphan hf-space-snapshot
    git lfs install --local >/dev/null
    cat > .gitattributes <<'EOF'
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
EOF
    git rm -r --cached -f --quiet .    # re-add everything so the LFS rules apply
    git add -A
    git commit --quiet -m "Deploy $SRC from $(git -C "$OLDPWD" rev-parse --abbrev-ref HEAD)"
    echo "Snapshot of $SRC: $(git ls-files | wc -l | tr -d ' ') files, in LFS:"
    git lfs ls-files --size | sed 's/^/  /'
    if [ -n "$DRY_RUN" ]; then
        echo "DRY_RUN set: not pushing."
    else
        echo "Pushing to $REMOTE (password = your Hugging Face write token) ..."
        git push --force "$REMOTE" hf-space-snapshot:main
        echo "Done. Build progress: the Space page -> Logs."
    fi
)
