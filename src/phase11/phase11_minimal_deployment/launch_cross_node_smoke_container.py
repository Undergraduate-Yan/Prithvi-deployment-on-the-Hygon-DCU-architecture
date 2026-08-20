#!/usr/bin/env python3
"""Host-side, immutable launcher for formal K100 cross-node smoke.

The locked static ``lsmod`` must be first on PATH before Docker starts the
container.  Setting PATH in the in-container wrapper is too late because the
image's LD_PRELOAD initializer can call ``lsmod`` before the entrypoint runs.
This launcher records the exact Docker argv and environment as a support
receipt next to (but never inside) the immutable bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "phase11_k100_container_launch_attestation_v1"
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
LSMOD_SIZE = 819_664
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"
CONTAINER_PATH = (
    "/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
)
PREENTRYPOINT_TOKEN = "docker_run_environment_before_entrypoint_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular non-symlink file: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def write_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def copy_new(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"container evidence is not a regular file: {source}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 << 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-image-id", required=True)
    parser.add_argument("--node-label", choices=("K100-2", "K100-3"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host-hyhal", type=Path, default=Path("/opt/hyhal"))
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--shm-size", default="8g")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utc_now()
    output = args.output_dir.absolute()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output.name) or output.name in {".", ".."}:
        raise RuntimeError("smoke output must have a safe normal leaf directory name")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite smoke evidence: {output}")
    output_parent = output.parent.resolve(strict=True)
    output = output_parent / output.name
    bundle_raw = args.bundle.absolute()
    if bundle_raw.is_symlink():
        raise RuntimeError("bundle must not be a symlink")
    preflight_bundle = bundle_raw.resolve(strict=True)
    if not preflight_bundle.is_dir():
        raise RuntimeError("bundle must be a directory")
    if (
        preflight_bundle == output_parent
        or preflight_bundle in output_parent.parents
        or output_parent in preflight_bundle.parents
    ):
        raise RuntimeError("smoke output parent and immutable bundle must be separate")
    # Create the evidence root on the host, before Docker, with O_EXCL-like
    # mkdir semantics.  Docker writes only a child directory, so root-owned
    # container files cannot prevent the invoking user from writing the outer
    # logs and attestation.
    output.mkdir(parents=False, exist_ok=False)
    output_stat = output.stat()
    stdout_path = output / "container_launcher_stdout.log"
    stderr_path = output / "container_launcher_stderr.log"
    formal_result = output / "result.json"
    formal_fingerprint = output / "target_runtime_fingerprint.json"
    container_smoke = output / "container_smoke"

    bundle = preflight_bundle
    requested_hyhal = args.host_hyhal.absolute()
    hyhal = None
    manifest = {}
    manifest_identity = None
    shim = None
    inspect_returncode = None
    observed_image = None
    command = None
    docker_returncode = None
    docker_stdout = ""
    docker_stderr = ""
    child_result = None
    child_fingerprint = None
    result_payload = None
    failure = None
    passed = False
    try:
        if args.device < 0:
            raise RuntimeError("device index must be non-negative")
        if Path(args.docker).name != "docker":
            raise RuntimeError("formal launcher requires the Docker CLI executable")
        # /opt/hyhal is commonly a host symlink.  Resolve it before constructing
        # the bind mount and record both requested and resolved paths.
        hyhal = requested_hyhal.resolve(strict=True)
        if not hyhal.is_dir():
            raise RuntimeError("resolved host hyhal mount must be a directory")
        if not IMAGE_PATTERN.fullmatch(args.official_image_id):
            raise RuntimeError("official image must be an immutable sha256 image ID")
        if not SHA_PATTERN.fullmatch(args.expected_manifest_sha256):
            raise RuntimeError("manifest SHA256 must be 64 lowercase hex digits")

        manifest_path = bundle / "manifest.json"
        manifest_identity = identity(manifest_path)
        if manifest_identity["sha256"] != args.expected_manifest_sha256:
            raise RuntimeError("detached manifest SHA256 drift")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_contract", {}).get("official_image_id") != args.official_image_id:
            raise RuntimeError("launcher image differs from locked bundle image")
        shim = identity(bundle / "tools/bin/lsmod")
        if (shim["size_bytes"], shim["sha256"]) != (LSMOD_SIZE, LSMOD_SHA256):
            raise RuntimeError("locked static lsmod identity drift")

        inspect = subprocess.run(
            [args.docker, "image", "inspect", "--format", "{{.Id}}", args.official_image_id],
            capture_output=True,
            text=True,
            check=False,
        )
        inspect_returncode = inspect.returncode
        observed_image = inspect.stdout.strip()
        if inspect.returncode != 0 or observed_image != args.official_image_id:
            raise RuntimeError(
                f"Docker image attestation failed: rc={inspect.returncode}, id={observed_image!r}"
            )

        command = [
            args.docker,
            "run",
            "--rm",
            "--ipc=host",
            f"--shm-size={args.shm_size}",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "-v",
            f"{bundle}:/bundle:ro",
            "-v",
            f"{output}:/acceptance",
            "-v",
            f"{hyhal}:/opt/hyhal:ro",
            "-e",
            f"PATH={CONTAINER_PATH}",
            "-e",
            f"PHASE11_K100_IMAGE_ID={args.official_image_id}",
            "-e",
            f"PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION={PREENTRYPOINT_TOKEN}",
            "--entrypoint",
            "/bundle/tools/run_cross_node_smoke.sh",
            args.official_image_id,
            "--bundle",
            "/bundle",
            "--expected-manifest-sha256",
            args.expected_manifest_sha256,
            "--output-dir",
            "/acceptance/container_smoke",
            "--device",
            str(args.device),
            "--node-label",
            args.node_label,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        docker_returncode = completed.returncode
        docker_stdout = completed.stdout
        docker_stderr = completed.stderr
        if container_smoke.is_symlink() or not container_smoke.is_dir():
            raise RuntimeError("container did not create a regular isolated smoke directory")
        child_result = container_smoke / "result.json"
        child_fingerprint = container_smoke / "target_runtime_fingerprint.json"
        if not child_result.is_file() or not child_fingerprint.is_file():
            raise RuntimeError("container smoke did not produce result and runtime fingerprint")
        copy_new(child_result, formal_result)
        copy_new(child_fingerprint, formal_fingerprint)
        child_result_identity = identity(child_result)
        formal_result_identity = identity(formal_result)
        if (
            child_result_identity["size_bytes"],
            child_result_identity["sha256"],
        ) != (
            formal_result_identity["size_bytes"],
            formal_result_identity["sha256"],
        ):
            raise RuntimeError("formal smoke receipt copy identity drift")
        child_fingerprint_identity = identity(child_fingerprint)
        formal_fingerprint_identity = identity(formal_fingerprint)
        if (
            child_fingerprint_identity["size_bytes"],
            child_fingerprint_identity["sha256"],
        ) != (
            formal_fingerprint_identity["size_bytes"],
            formal_fingerprint_identity["sha256"],
        ):
            raise RuntimeError("formal runtime fingerprint copy identity drift")
        result_payload = json.loads(formal_result.read_text(encoding="utf-8"))
        passed = bool(
            docker_returncode == 0
            and isinstance(result_payload, dict)
            and result_payload.get("status") == "passed"
        )
        if not passed:
            raise RuntimeError("container model smoke receipt did not pass")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}

    # The host owns output/, so these writes remain possible even when the
    # container-created child tree is owned by root.
    write_new(stdout_path, docker_stdout)
    write_new(stderr_path, docker_stderr if failure is None else docker_stderr + f"\nlauncher: {failure['type']}: {failure['message']}\n")
    stdout_path = output / "container_launcher_stdout.log"
    stderr_path = output / "container_launcher_stderr.log"
    receipt = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "generator": identity(Path(__file__)),
        "hostname": platform.node(),
        "node_label": args.node_label,
        "device": args.device,
        "bundle_id": manifest.get("bundle_id") if isinstance(manifest, dict) else None,
        "manifest": manifest_identity,
        "official_image_id": args.official_image_id,
        "docker_image_inspect": {
            "returncode": inspect_returncode,
            "observed_image_id": observed_image,
        },
        "docker_argv": command,
        "docker_returncode": docker_returncode,
        "container_bundle_dir": "/bundle",
        "container_output_dir": "/acceptance/container_smoke",
        "host_output_dir": str(output),
        "host_output_ownership": {
            "created_exclusively_before_docker": True,
            "uid": getattr(output_stat, "st_uid", None),
            "gid": getattr(output_stat, "st_gid", None),
            "mode": oct(output_stat.st_mode & 0o777),
        },
        "launcher_logs": {
            "stdout": identity(stdout_path),
            "stderr": identity(stderr_path),
        },
        "pre_entrypoint_environment": {
            "PATH": CONTAINER_PATH,
            "PHASE11_K100_IMAGE_ID": args.official_image_id,
            "PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION": PREENTRYPOINT_TOKEN,
        },
        "static_lsmod": {
            "container_path": "/bundle/tools/bin/lsmod",
            "size_bytes": shim["size_bytes"] if shim else None,
            "sha256": shim["sha256"] if shim else None,
        },
        "host_hyhal_mount": {
            "requested_source": str(requested_hyhal),
            "source": str(hyhal) if hyhal else None,
            "target": "/opt/hyhal",
            "read_only": True,
            "requested_source_was_symlink": requested_hyhal.is_symlink(),
            "symlink_resolved_before_docker": hyhal is not None,
        },
        "container_smoke_result": identity(child_result) if child_result and child_result.is_file() else None,
        "container_runtime_fingerprint": (
            identity(child_fingerprint) if child_fingerprint and child_fingerprint.is_file() else None
        ),
        "smoke_result": identity(formal_result) if formal_result.is_file() else None,
        "formal_runtime_fingerprint": (
            identity(formal_fingerprint) if formal_fingerprint.is_file() else None
        ),
        "claims": {
            "path_injected_by_docker_run_before_entrypoint": True,
            "locked_static_lsmod_first_on_path": True,
            "host_hyhal_mounted_read_only": True,
            "host_output_created_exclusively_before_docker": True,
            "container_writes_isolated_subdirectory": True,
            "formal_receipts_copied_without_byte_drift": passed,
            "invalid_prior_launcher_runs_count_toward_acceptance": False,
            "no_hyhal_diagnostics_count_toward_acceptance": False,
        },
        "failure": failure,
    }
    attestation = output / "container_launch_attestation.json"
    if attestation.exists():
        raise RuntimeError("refusing to overwrite launch attestation")
    write_new(
        attestation,
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
