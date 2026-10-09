#!/usr/bin/env python3
"""Build the local CPU source archive from the immutable v2.0.0 Git tag.

Only explicit CPU/test/provenance inputs are shipped. This script writes
build artifacts under dist/, never edits a source file or publishes remotely.
Run bin/verify_release.jl in the extracted tree before distributing it.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent.parent
TAG = "v2.0.0"
VERSION = "2.0.0"
PREFIX = f"KTrader-{VERSION}"
ROOT_FILES = {
    "Project.toml", "Manifest.toml", "README.md", "AGENTS.md",
    "CHANGELOG.md", "RELEASE.toml", "RELEASE_CPU_2_0.md", "ROADMAP_2_1.md",
    "universe.txt",
}
TEST_LIBRARIES = {
    "dev/probes.jl", "dev/m1_artifact_replay.jl", "dev/m1_replay_manifest.toml",
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=15)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def safe_name(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and ".." not in p.parts and "\\" not in path


def main() -> None:
    require(len(os.sys.argv) == 1, "this release builder takes no arguments")
    require(git("cat-file", "-t", f"refs/tags/{TAG}").strip() == b"tag", "an annotated Final tag is required")
    commit = git("rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}").decode().strip()
    timestamp = int(git("show", "-s", "--format=%ct", commit))
    tracked = {p for p in git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0") if p}

    def blob(path: str) -> bytes:
        require(safe_name(path) and path in tracked, f"missing or unsafe tagged input: {path}")
        return git("show", f"{commit}:{path}")

    metadata = tomllib.loads(blob("RELEASE.toml").decode())
    require(metadata["version"] == VERSION and metadata["status"] == "Final", "not the Final release")
    evidence = metadata["acceptance"]
    acceptance_bytes = blob(evidence["path"])
    qualification_bytes = blob(evidence["qualification_path"])
    require(digest(acceptance_bytes) == evidence["sha256"], "acceptance bytes changed")
    require(digest(qualification_bytes) == evidence["qualification_sha256"], "qualification bytes changed")
    acceptance = tomllib.loads(acceptance_bytes.decode())
    qualification = tomllib.loads(qualification_bytes.decode())

    selected = ROOT_FILES | TEST_LIBRARIES
    selected |= {p for p in tracked if p.startswith(("src/", "test/", "bin/", "release/"))}
    selected |= {evidence["path"], evidence["qualification_path"]}
    selected |= set(acceptance["evidence_sha256"])
    # Ship actual group logs with receipts; not the unrelated full dev history.
    qdir = str(PurePosixPath(evidence["qualification_path"]).parent)
    selected |= {f"{qdir}/{group}.log" for group in qualification["groups"]}
    selected |= {f"{qdir}/status_complete.log"}
    selected |= {
        "dev/evidence/cpu20_acceptance_20261009/audit.log",
        "dev/evidence/cpu20_acceptance_20261009/status.log",
        "dev/evidence/final_2_0_0_20261009/cpu_acceptance_before_version.log",
        "dev/evidence/final_2_0_0_20261009/release_verifier_tests.log",
        "dev/evidence/final_2_0_0_20261009/package_smoke.log",
        "dev/evidence/final_2_0_0_20261009/final_source_verification.log",
    }
    require(selected <= tracked, "release allowlist contains uncommitted/missing files: " + repr(sorted(selected - tracked)))
    require(all(safe_name(p) for p in selected), "unsafe release path")
    require(not any(p.startswith(("data/", ".git/", ".wanxiangshu/")) or
                    (p.startswith("dev/") and (p.endswith((".jls", ".bin", ".cpp")) or "hip_core" in p))
                    for p in selected), "private/GPU payload entered CPU release")

    # Git supplies only immutable tagged bytes, not a moving working tree.
    raw = git("archive", "--format=tar", commit, "--", *sorted(selected))
    files: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if member.isdir():
                continue
            require(member.isfile() and member.name in selected, f"unexpected archive entry: {member.name}")
            require(member.name not in files, "duplicate source archive entry")
            stream = archive.extractfile(member)
            require(stream is not None, "missing archive bytes")
            files[member.name] = (stream.read(), 0o755 if member.mode & 0o111 else 0o644)
    require(set(files) == selected, "Git archive did not contain the exact allowlist")
    for path, expected in acceptance["evidence_sha256"].items():
        require(digest(files[path][0]) == expected, f"evidence identity changed: {path}")
    for path, expected in qualification["sources"].items():
        if path.startswith(("src/", "test/", "bin/")) or path in TEST_LIBRARIES:
            require(path in files and digest(files[path][0]) == expected, f"qualified source changed: {path}")

    build = (f'version = "{VERSION}"\nstatus = "Final"\ntag = "{TAG}"\n'
             f'commit = "{commit}"\nsource = "immutable local Git tag"\ngpu_included = false\n')
    files["release/BUILD.toml"] = (build.encode(), 0o644)
    checksum_text = "".join(f"{digest(data)}  {name}\n" for name, (data, _) in sorted(files.items()))
    files["release/SHA256SUMS"] = (checksum_text.encode(), 0o644)

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    output = dist / f"{PREFIX}.tar.gz"
    sums = dist / f"{PREFIX}.tar.gz.sha256"
    receipt = dist / f"{PREFIX}.release.json"
    require(not any(p.exists() for p in (output, sums, receipt)), "refusing to overwrite release artifacts")
    # Generate completely before publishing the path. Existing release files
    # are never replaced, even if another build races this one.
    with tempfile.NamedTemporaryFile(dir=dist, prefix=".package-", delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=temp, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, (data, mode) in sorted(files.items()):
                        entry = tarfile.TarInfo(f"{PREFIX}/{name}")
                        entry.size, entry.mode, entry.mtime = len(data), mode, timestamp
                        entry.uid = entry.gid = 0
                        entry.uname = entry.gname = ""
                        archive.addfile(entry, io.BytesIO(data))
            temp.flush()
            os.fsync(temp.fileno())
            os.link(temporary, output)  # exclusive: fails if the target exists
        finally:
            temporary.unlink(missing_ok=True)  # only this builder's temporary file
    archive_hash = digest(output.read_bytes())
    with sums.open("x") as stream:
        stream.write(f"{archive_hash}  {output.name}\n")
    info = {"version": VERSION, "status": "Final", "tag": TAG, "commit": commit,
            "archive": output.name, "sha256": archive_hash, "bytes": output.stat().st_size,
            "payload_files": len(files), "market_data_included": False, "gpu_included": False,
            "remote_published": False, "extracted_source_validation": "required; see validation log"}
    with receipt.open("x") as stream:
        json.dump(info, stream, indent=2, sort_keys=True)
        stream.write("\n")

    # Create a disposable extraction for the Julia verifier/tests. No shell,
    # arbitrary path extraction, symlink following or workspace writes.
    verification = Path(tempfile.mkdtemp(prefix=f"verify-{VERSION}-", dir=dist)) / PREFIX
    verification.mkdir()
    with tarfile.open(output, mode="r:gz") as archive:
        entries = list(archive)
        require(len(entries) == len(files), "wrong packaged file count")
        for entry in entries:
            require(entry.isfile() and entry.name.startswith(PREFIX + "/"), "invalid output tar member")
            relative = entry.name[len(PREFIX) + 1:]
            require(relative in files and safe_name(relative), "unexpected output member")
            data = archive.extractfile(entry).read()
            require(data == files[relative][0], f"packaging changed bytes: {relative}")
            path = verification.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(entry.mode)
    print(json.dumps({**info, "verification_root": str(verification.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
