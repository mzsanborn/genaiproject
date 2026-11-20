"""
Off-machine backup of training checkpoints and results.

Training reads and writes ONLY the local disk (fast, reliable). This module copies
selected files to persistent storage with an upload that returns only once the
remote side has the file, so "backed up" really means backed up. It replaces
writing into a mounted Google Drive folder: on Colab that mount uploads
asynchronously and can silently stop uploading (we lost ~3 h of checkpoints that
way), and its glitches crashed the training loop.

Storage use: each upload creates a new remote file and then PERMANENTLY deletes the
older copies of the same name (no Trash, no revision history), so only the latest
version of each file is kept.

Backends
  drive  Google Drive through the Drive v3 API. On Colab, run
         `from google.colab import auth; auth.authenticate_user()` in the notebook
         first; the training subprocess picks the credentials up from there.
  local  A directory (used for tests, or any synced/mounted folder you trust).
"""
import logging
import os
import shutil
import threading
import time


# --------------------------------------------------------------------------- backends
class LocalDirBackend:
    def __init__(self, root):
        self.root = root

    def describe(self):
        return f"local dir {self.root}"

    def upload(self, local_path, remote_rel):
        dst = os.path.join(self.root, remote_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".uploading"
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, dst)

    def download(self, remote_rel, local_path):
        src = os.path.join(self.root, remote_rel)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copyfile(src, local_path)
        return True


class DriveApiBackend:
    """Files live under My Drive/<root>/<remote_rel>."""
    FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, root):
        import google.auth
        from googleapiclient.discovery import build
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.root = root.strip("/")
        self._folders = {}

    def describe(self):
        return f"Google Drive (API): My Drive/{self.root}"

    @staticmethod
    def _q(s):
        return s.replace("\\", "\\\\").replace("'", "\\'")

    def _children(self, parent, name, folder=None):
        q = f"'{parent}' in parents and name = '{self._q(name)}' and trashed = false"
        if folder is True:
            q += f" and mimeType = '{self.FOLDER}'"
        elif folder is False:
            q += f" and mimeType != '{self.FOLDER}'"
        res = self.svc.files().list(q=q, spaces="drive", orderBy="modifiedTime desc",
                                    fields="files(id, name, modifiedTime, size)").execute()
        return res.get("files", [])

    def _folder_id(self, rel_dir, create):
        parts = [p for p in f"{self.root}/{rel_dir}".split("/") if p]
        parent, path = "root", ""
        for p in parts:
            path = f"{path}/{p}"
            if path not in self._folders:
                found = self._children(parent, p, folder=True)
                if found:
                    self._folders[path] = found[0]["id"]
                elif create:
                    meta = {"name": p, "mimeType": self.FOLDER, "parents": [parent]}
                    self._folders[path] = self.svc.files().create(body=meta, fields="id").execute()["id"]
                else:
                    return None
            parent = self._folders[path]
        return parent

    def upload(self, local_path, remote_rel):
        from googleapiclient.http import MediaFileUpload
        rel_dir, name = os.path.split(remote_rel)
        parent = self._folder_id(rel_dir, create=True)
        old = self._children(parent, name, folder=False)
        media = MediaFileUpload(local_path, mimetype="application/octet-stream",
                                resumable=True, chunksize=64 * 1024 * 1024)
        req = self.svc.files().create(body={"name": name, "parents": [parent]},
                                      media_body=media, fields="id, size")
        resp = None
        while resp is None:
            _, resp = req.next_chunk(num_retries=5)
        if int(resp.get("size", -1)) != os.path.getsize(local_path):
            raise IOError(f"size mismatch after uploading {remote_rel}: {resp.get('size')}")
        # new copy is safely stored -> permanently delete the older ones (frees quota now)
        for f in old:
            self.svc.files().delete(fileId=f["id"]).execute()

    def download(self, remote_rel, local_path):
        from googleapiclient.http import MediaIoBaseDownload
        rel_dir, name = os.path.split(remote_rel)
        parent = self._folder_id(rel_dir, create=False)
        if parent is None:
            return False
        found = self._children(parent, name, folder=False)
        if not found:
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        tmp = local_path + ".downloading"
        with open(tmp, "wb") as fh:
            dl = MediaIoBaseDownload(fh, self.svc.files().get_media(fileId=found[0]["id"]),
                                     chunksize=64 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk(num_retries=5)
        os.replace(tmp, local_path)
        return True


def make_backend(kind, root):
    if kind == "drive":
        return DriveApiBackend(root)
    if kind == "local":
        return LocalDirBackend(root)
    raise ValueError(f"unknown backup backend {kind}")


# --------------------------------------------------------------------------- manager
class Backup:
    """
    Periodic, non-blocking checkpoint backup.

      maybe_backup()  called after every local checkpoint; at most every `every_min`
                      minutes it snapshots the checkpoint files (local copy, a few
                      seconds) and uploads them in a background thread, so training
                      keeps going during the upload.
      flush()         blocking upload of the current checkpoint (end of training).
      sync_tree()     blocking upload of result files (samples, metrics, logs).
      restore()       download checkpoint files into a fresh machine before resuming.
    """
    CKPT_FILES = ("ckpt.pth", "ckpt_extra.pth", "ckpt_best.pth", "val_loss.csv", "loss.csv", "config.yml")

    def __init__(self, backend, exp_dir, every_min=20.0):
        self.backend = backend
        self.exp_dir = os.path.abspath(exp_dir)
        self.every = every_min * 60
        self.last_start = 0.0
        self.thread = None
        self.snap_dir = os.path.join(self.exp_dir, ".backup_snapshot")

    def describe(self):
        return self.backend.describe()

    def _rel(self, path):
        return os.path.relpath(os.path.abspath(path), self.exp_dir).replace(os.sep, "/")

    def check(self):
        """Fail fast (before training) if the backend is unusable, e.g. not authenticated."""
        os.makedirs(self.exp_dir, exist_ok=True)
        probe = os.path.join(self.exp_dir, "backup_heartbeat.txt")
        with open(probe, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        self.backend.upload(probe, "backup_heartbeat.txt")

    def _upload_all(self, pairs, label):
        t0 = time.time()
        for local, rel in pairs:
            for attempt in range(4):
                try:
                    self.backend.upload(local, rel)
                    break
                except Exception as e:  # network errors etc.
                    if attempt == 3:
                        logging.error(f"[backup] {label}: upload of {rel} FAILED: {e!r}")
                        return False
                    logging.warning(f"[backup] {label}: upload of {rel} failed ({e!r}); retrying")
                    time.sleep(30 * (attempt + 1))
        logging.info(f"[backup] {label}: uploaded {len(pairs)} file(s) in {time.time() - t0:.0f}s")
        return True

    def _ckpt_pairs(self, log_path, snapshot):
        pairs = []
        if snapshot:
            shutil.rmtree(self.snap_dir, ignore_errors=True)
            os.makedirs(self.snap_dir)
        for name in self.CKPT_FILES:
            src = os.path.join(log_path, name)
            if not os.path.exists(src):
                continue
            local = src
            if snapshot:  # freeze a copy so training can overwrite the original meanwhile
                local = os.path.join(self.snap_dir, name)
                shutil.copyfile(src, local)
            pairs.append((local, self._rel(src)))
        # ckpt_extra.pth after ckpt.pth is fine: resume checks they match (see runner)
        return pairs

    def maybe_backup(self, log_path, step):
        now = time.time()
        if now - self.last_start < self.every:
            return
        if self.thread is not None and self.thread.is_alive():
            logging.warning("[backup] previous upload still running; skipping this checkpoint")
            return
        self.last_start = now
        pairs = self._ckpt_pairs(log_path, snapshot=True)
        self.thread = threading.Thread(target=self._upload_all, args=(pairs, f"step {step}"), daemon=True)
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()

    def flush(self, log_path, label="final checkpoint"):
        self.wait()
        return self._upload_all(self._ckpt_pairs(log_path, snapshot=False), label)

    def sync_tree(self, root, label="results", skip=("ckpt.pth", "ckpt_extra.pth", "ckpt_best.pth")):
        self.wait()
        pairs = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith(".")]
            for f in files:
                if f not in skip and not f.endswith((".tmp", ".uploading", ".downloading")):
                    p = os.path.join(d, f)
                    pairs.append((p, self._rel(p)))
        return self._upload_all(pairs, label)

    def restore(self, log_path):
        """Download the backed-up checkpoint files into log_path. Returns True if ckpt.pth was restored."""
        os.makedirs(log_path, exist_ok=True)
        got = []
        for name in self.CKPT_FILES:
            local = os.path.join(log_path, name)
            if self.backend.download(self._rel(local), local):
                got.append(name)
        logging.info(f"[backup] restored from {self.describe()}: {got or 'nothing'}")
        return "ckpt.pth" in got
