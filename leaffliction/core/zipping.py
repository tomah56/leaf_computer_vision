#!/usr/bin/env python3
"""Create a zip archive from provided paths and generate signature.txt.

The script zips all given files/directories, computes SHA1 for the zip,
and writes a `signature.txt` file at the Git repository root by default.

Example:
	python3 leaffliction/core/zipping.py \
		--output dataset_bundle.zip \
		leaffliction/images leaffliction/model.pth
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


def sha1_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
	"""Return SHA1 hash for a file."""
	hasher = hashlib.sha1()
	with file_path.open("rb") as file_obj:
		while True:
			chunk = file_obj.read(chunk_size)
			if not chunk:
				break
			hasher.update(chunk)
	return hasher.hexdigest()


def iter_directory_files(directory: Path) -> Iterable[Path]:
	"""Yield all files from a directory in deterministic sorted order."""
	files = [path for path in directory.rglob("*") if path.is_file()]
	files.sort(key=lambda path: str(path.relative_to(directory)).replace("\\", "/"))
	return files


def sha1_directory(directory: Path) -> str:
	"""Return a deterministic SHA1 hash of directory structure + file content."""
	hasher = hashlib.sha1()
	for file_path in iter_directory_files(directory):
		relative_path = file_path.relative_to(directory)
		normalized = str(relative_path).replace("\\", "/")
		hasher.update(normalized.encode("utf-8"))
		hasher.update(b"\0")
		with file_path.open("rb") as file_obj:
			for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
				hasher.update(chunk)
		hasher.update(b"\0")
	return hasher.hexdigest()


def find_git_root(start: Path) -> Path:
	"""Find repository root by searching parents for a .git entry."""
	current = start.resolve()
	for candidate in [current, *current.parents]:
		if (candidate / ".git").exists():
			return candidate
	raise FileNotFoundError("No Git repository root found from current path.")


def add_path_to_zip(zip_file: ZipFile, input_path: Path) -> None:
	"""Add a file or directory to zip preserving its top-level name."""
	if input_path.is_file():
		zip_file.write(input_path, arcname=input_path.name)
		return

	parent = input_path.parent
	for file_path in iter_directory_files(input_path):
		arcname = file_path.relative_to(parent)
		zip_file.write(file_path, arcname=str(arcname).replace("\\", "/"))


def create_zip(output_zip: Path, input_paths: list[Path]) -> None:
	"""Create zip archive with all input paths."""
	output_zip.parent.mkdir(parents=True, exist_ok=True)
	with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zip_file:
		for input_path in input_paths:
			add_path_to_zip(zip_file, input_path)


def write_signature_file(
	signature_file: Path,
	output_zip: Path,
	zip_sha1: str,
) -> None:
	"""Write signature.txt with only the zip SHA1 signature line."""
	signature_file.write_text(
		f"{zip_sha1} {output_zip.name}\n",
		encoding="utf-8",
	)


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(
		description=(
			"Zip provided files/directories and generate signature.txt "
			"with SHA1 signatures."
		)
	)
	parser.add_argument(
		"inputs",
		nargs="+",
		help="Files/directories to include in the zip archive.",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="directory.zip",
		help="Output zip filename (default: directory.zip).",
	)
	parser.add_argument(
		"-s",
		"--signature",
		default=None,
		help=(
			"Path to signature file. "
			"Default: <git-root>/signature.txt"
		),
	)
	return parser.parse_args()


def main() -> int:
	"""Run zipping + signature generation."""
	args = parse_args()

	raw_paths = [Path(path).expanduser().resolve() for path in args.inputs]
	missing = [str(path) for path in raw_paths if not path.exists()]
	if missing:
		print("Error: these input paths do not exist:")
		for path in missing:
			print(f"- {path}")
		return 1

	output_zip = Path(args.output).expanduser().resolve()
	create_zip(output_zip, raw_paths)
	zip_sha1 = sha1_file(output_zip)

	if args.signature:
		signature_file = Path(args.signature).expanduser().resolve()
	else:
		signature_file = find_git_root(Path.cwd()) / "signature.txt"

	write_signature_file(signature_file, output_zip, zip_sha1)

	print(f"Created zip: {output_zip}")
	print(f"SHA1: {zip_sha1} {output_zip.name}")
	print(f"Created signature: {signature_file}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

