"""Output-directory policy and atomic run manifests."""
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid

import torch


MANIFEST_NAME = 'run.json'
SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunManager:
    """Own one run's output policy and its main-process-only manifest."""

    def __init__(
        self,
        config,
        *,
        total,
        partitions,
        repo_root=None,
    ):
        if type(total) is not int or total < 1:
            raise ValueError("total work items must be a positive integer")

        self.config = config
        self.total = total
        self.repo_root = Path(
            repo_root
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        ).resolve()
        raw_fig_root = self.repo_root / 'fig'
        if raw_fig_root.is_symlink():
            raise ValueError(
                f"Top-level fig directory may not be a symbolic link: "
                f"{raw_fig_root}"
            )
        self.fig_root = raw_fig_root.resolve()
        self.output_dir = self._resolve_output_dir(config.get_output_dir())
        self.manifest_path = self.output_dir / MANIFEST_NAME
        self.lock_path = self.output_dir.parent / (
            f'.{self.output_dir.name}.lock'
        )
        self._lock_fd = None
        self._terminal_recorded = False
        self.partitions = self._normalize_partitions(partitions)
        self.gpu_ids = [item['gpu_id'] for item in self.partitions]
        self.expected = self._build_expected()
        self.manifest = None

    def _resolve_output_dir(self, configured_path):
        raw_path = Path(configured_path)
        if not raw_path.is_absolute():
            raw_path = self.repo_root / raw_path
        if raw_path.is_symlink():
            raise ValueError(
                f"Output directory may not be a symbolic link: {raw_path}"
            )

        output_dir = raw_path.resolve()
        try:
            relative = output_dir.relative_to(self.fig_root)
        except ValueError as error:
            raise ValueError(
                f"Output directory must be inside {self.fig_root}: "
                f"{output_dir}"
            ) from error
        if not relative.parts:
            raise ValueError("Refusing to use the top-level fig directory")
        return output_dir

    def _normalize_partitions(self, partitions):
        normalized = []
        cursor = 0
        for item in partitions:
            required = {'gpu_id', 'start', 'end', 'worker_seed'}
            if set(item) != required:
                raise ValueError(
                    f"Partition keys must be {sorted(required)}"
                )
            normalized_item = {
                'gpu_id': int(item['gpu_id']),
                'start': int(item['start']),
                'end': int(item['end']),
                'worker_seed': int(item['worker_seed']),
            }
            if normalized_item['start'] != cursor:
                raise ValueError("Partitions must be ordered and contiguous")
            if normalized_item['end'] <= normalized_item['start']:
                raise ValueError("Every partition must contain work")
            normalized.append(normalized_item)
            cursor = normalized_item['end']

        if cursor != self.total or not normalized:
            raise ValueError(
                f"Partitions cover [0, {cursor}), expected [0, {self.total})"
            )
        return normalized

    def _build_expected(self):
        if self.config.mode == 'single':
            results = [
                f'{index}.png'
                for index in range(self.config.num_samples)
            ]
            common = ['label.png', 'input.png']
        else:
            results = [
                f'{index:03d}_{sample}.png'
                for index in range(self.total)
                for sample in range(self.config.num_samples)
            ]
            common = [
                *[
                    f'labels/{index:03d}.png'
                    for index in range(self.total)
                ],
                *[
                    f'inputs/{index:03d}.png'
                    for index in range(self.total)
                ],
            ]
        return {'results': results, 'common': common}

    @property
    def topology(self):
        return {
            'gpu_ids': self.gpu_ids,
            'partitions': self.partitions,
            'seed_scheme': (
                'worker_seed = (config.seed + partition.start) mod 2**63'
            ),
        }

    def prepare(self, *, overwrite=False, resume=False):
        """Prepare the exact output directory and mark the run as running."""
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")

        self.fig_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            if overwrite and self.output_dir.exists():
                if not self.output_dir.is_dir():
                    raise ValueError(
                        f"Output path exists but is not a directory: "
                        f"{self.output_dir}"
                    )
                shutil.rmtree(self.output_dir)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            nonempty = any(self.output_dir.iterdir())

            if resume:
                if not self.manifest_path.is_file():
                    raise FileNotFoundError(
                        "Cannot resume an output directory without run.json; "
                        "use --overwrite for legacy or unrelated outputs"
                    )
                existing = self._read_manifest()
                self._validate_resume(existing)
                self.manifest = existing
                self.manifest['status'] = 'running'
                self.manifest['attempts'] = (
                    int(self.manifest.get('attempts', 0)) + 1
                )
                self.manifest['error'] = None
                self.manifest['timestamps']['started_at'] = _utc_now()
                self.manifest['timestamps']['completed_at'] = None
                self.manifest['timestamps']['failed_at'] = None
                self.manifest['last_runtime'] = self._collect_runtime()
            else:
                if nonempty:
                    raise FileExistsError(
                        f"Output directory is not empty: {self.output_dir}. "
                        "Use --overwrite to replace it or --resume for a "
                        "matching manifest-backed run."
                    )
                now = _utc_now()
                self.manifest = {
                    'schema_version': SCHEMA_VERSION,
                    'status': 'running',
                    'config_fingerprint': self.config.config_fingerprint,
                    'config': self.config.run_spec(),
                    'output_dir': self.output_dir.relative_to(
                        self.repo_root
                    ).as_posix(),
                    'topology': self.topology,
                    'expected': self.expected,
                    'observed': {'generated': [], 'missing': []},
                    'attempts': 1,
                    'timestamps': {
                        'created_at': now,
                        'started_at': now,
                        'completed_at': None,
                        'failed_at': None,
                    },
                    'provenance': self._collect_provenance(),
                    'runtime': self._collect_runtime(),
                    'error': None,
                }

            self._refresh_observed()
            self._write_manifest()
            self._terminal_recorded = False
            return bool(self.manifest['observed']['missing'])
        except BaseException:
            try:
                self.release()
            except Exception as release_error:
                print(
                    f"Additionally failed to release the run lock: "
                    f"{release_error}",
                    file=sys.stderr,
                )
            raise

    def _acquire_lock(self):
        if self._lock_fd is not None:
            raise RuntimeError("Run lock is already held")

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            lock_fd = os.open(self.lock_path, flags, 0o600)
        except FileExistsError as error:
            try:
                details = self.lock_path.read_text(
                    encoding='utf-8'
                ).strip()
            except OSError:
                details = 'unreadable lock metadata'
            raise FileExistsError(
                f"Another process holds the run lock {self.lock_path}. "
                f"Lock metadata: {details}. If the recorded process is no "
                "longer alive, inspect and remove this stale lock manually."
            ) from error

        self._lock_fd = lock_fd
        lock_metadata = {
            'hostname': platform.node(),
            'pid': os.getpid(),
            'created_at': _utc_now(),
            'config_fingerprint': self.config.config_fingerprint,
        }
        try:
            payload = (
                json.dumps(lock_metadata, sort_keys=True) + '\n'
            ).encode('utf-8')
            remaining = memoryview(payload)
            while remaining:
                written = os.write(lock_fd, remaining)
                if written == 0:
                    raise OSError("Could not write run-lock metadata")
                remaining = remaining[written:]
            os.fsync(lock_fd)
        except BaseException:
            self.release()
            raise

    def release(self):
        """Release only the lock inode acquired by this manager."""
        if self._lock_fd is None:
            return

        lock_fd = self._lock_fd
        self._lock_fd = None
        try:
            owned_inode = os.fstat(lock_fd).st_ino
            try:
                path_inode = self.lock_path.stat().st_ino
            except FileNotFoundError:
                path_inode = None
            if path_inode == owned_inode:
                self.lock_path.unlink()
        finally:
            os.close(lock_fd)

    def _validate_resume(self, existing):
        if existing.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported run.json schema: "
                f"{existing.get('schema_version')!r}"
            )
        if existing.get('status') not in {
            'running',
            'completed',
            'failed',
        }:
            raise ValueError(
                f"Unsupported run status: {existing.get('status')!r}"
            )
        expected_output_dir = self.output_dir.relative_to(
            self.repo_root
        ).as_posix()
        if existing.get('output_dir') != expected_output_dir:
            raise ValueError("run.json output directory does not match")
        if existing.get('config_fingerprint') != (
            self.config.config_fingerprint
        ):
            stored_config = existing.get('config', {})
            stored_seed = stored_config.get('seed')
            stored_strict = stored_config.get('strict_deterministic')
            raise ValueError(
                "Resume config fingerprint does not match. "
                f"Stored execution options include seed={stored_seed!r} "
                f"and strict_deterministic={stored_strict!r}. Use the "
                "same --seed, the same presence of "
                "--strict-deterministic, and the same algorithm options."
            )
        if existing.get('config') != self.config.run_spec():
            raise ValueError(
                "Resume config differs despite matching short fingerprint"
            )
        if existing.get('topology') != self.topology:
            raise ValueError(
                "Resume requires the same active GPUs and work partitions"
            )
        if existing.get('expected') != self.expected:
            raise ValueError("Resume expected-file set does not match")
        if not isinstance(existing.get('timestamps'), dict):
            raise ValueError("Malformed timestamps in run.json")

    def complete(self):
        """Verify every declared output before marking the run completed."""
        self._require_manifest()
        try:
            self._refresh_observed()
            missing = self.manifest['observed']['missing']
            if missing:
                raise RuntimeError(
                    f"Run finished without {len(missing)} expected files; "
                    f"first missing: {missing[0]}"
                )

            self.manifest['status'] = 'completed'
            self.manifest['error'] = None
            self.manifest['timestamps']['completed_at'] = _utc_now()
            self.manifest['timestamps']['failed_at'] = None
            self._write_manifest()
            self._terminal_recorded = True
        except BaseException as error:
            try:
                try:
                    self._refresh_observed()
                    self._set_failure(error)
                    self._write_manifest()
                    self._terminal_recorded = True
                except Exception as manifest_error:
                    print(
                        "Additionally failed to record the run failure in "
                        f"run.json: {manifest_error}",
                        file=sys.stderr,
                    )
            finally:
                try:
                    self.release()
                except Exception as release_error:
                    print(
                        f"Additionally failed to release the run lock: "
                        f"{release_error}",
                        file=sys.stderr,
                    )
            raise
        else:
            self.release()

    def fail(self, error):
        """Record a failed attempt without hiding the original exception."""
        if self._terminal_recorded:
            self.release()
            return
        try:
            if self.manifest is None:
                return
            if self._lock_fd is None:
                self._acquire_lock()
            self._refresh_observed()
            self._set_failure(error)
            self._write_manifest()
            self._terminal_recorded = True
        finally:
            self.release()

    def _set_failure(self, error):
        self.manifest['status'] = 'failed'
        self.manifest['timestamps']['failed_at'] = _utc_now()
        self.manifest['timestamps']['completed_at'] = None
        self.manifest['error'] = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': ''.join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            ),
        }

    def _refresh_observed(self):
        generated = []
        missing = []
        for relative_path in (
            self.expected['results'] + self.expected['common']
        ):
            if (self.output_dir / relative_path).is_file():
                generated.append(relative_path)
            else:
                missing.append(relative_path)
        self.manifest['observed'] = {
            'generated': generated,
            'missing': missing,
        }

    def _read_manifest(self):
        try:
            with self.manifest_path.open('r', encoding='utf-8') as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot read valid run.json at {self.manifest_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError("run.json must contain a JSON object")
        return manifest

    def _write_manifest(self):
        self._require_manifest()
        temporary = self.output_dir / (
            f'.{MANIFEST_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
        )
        try:
            with temporary.open('x', encoding='utf-8') as handle:
                json.dump(
                    self.manifest,
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _require_manifest(self):
        if self.manifest is None:
            raise RuntimeError("RunManager.prepare() has not completed")

    def _collect_provenance(self):
        def git(*args, preserve_output=False):
            result = subprocess.run(
                ['git', *args],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            if preserve_output:
                return result.stdout
            return result.stdout.strip()

        try:
            commit = git('rev-parse', 'HEAD')
            branch = git('branch', '--show-current')
            # Porcelain status begins with a two-column state. Do not call
            # str.strip() here: an unstaged change intentionally starts with
            # a space (for example, " M README.md"), and removing it changes
            # README.md into EADME.md when the path is sliced at column 3.
            status = git(
                'status',
                '--porcelain=v1',
                '--untracked-files=all',
                preserve_output=True,
            ).rstrip('\r\n')
            dirty_files = [
                line[3:]
                for line in status.splitlines()
                if len(line) >= 4
            ]
            git_info = {
                'commit': commit,
                'branch': branch,
                'dirty': bool(status),
                'dirty_files': dirty_files,
            }
        except (OSError, subprocess.CalledProcessError) as error:
            git_info = {
                'commit': None,
                'branch': None,
                'dirty': None,
                'dirty_files': [],
                'error': str(error),
            }

        return {
            'git': git_info,
            'command': sys.argv,
            'cwd': str(Path.cwd()),
            'hostname': platform.node(),
        }

    def _collect_runtime(self):
        packages = {}
        for package in (
            'numpy',
            'Pillow',
            'PyYAML',
            'scipy',
            'scikit-image',
            'deepinv',
        ):
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = None

        gpu_names = {}
        for gpu_id in self.gpu_ids:
            try:
                gpu_names[str(gpu_id)] = torch.cuda.get_device_name(gpu_id)
            except Exception as error:
                gpu_names[str(gpu_id)] = f'unavailable: {error}'

        warn_only = None
        warn_only_query = getattr(
            torch,
            'is_deterministic_algorithms_warn_only_enabled',
            None,
        )
        if warn_only_query is not None:
            warn_only = warn_only_query()

        return {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'cudnn': (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            'packages': packages,
            'gpu_names': gpu_names,
            'determinism': {
                'strict_requested': self.config.strict_deterministic,
                'algorithms_enabled': (
                    torch.are_deterministic_algorithms_enabled()
                ),
                'warn_only': warn_only,
                'cudnn_deterministic': (
                    torch.backends.cudnn.deterministic
                ),
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cublas_workspace_config': os.environ.get(
                    'CUBLAS_WORKSPACE_CONFIG'
                ),
            },
        }
