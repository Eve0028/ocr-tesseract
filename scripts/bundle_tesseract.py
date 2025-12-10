#!/usr/bin/env python3
"""Copy required Tesseract files into the project for PyInstaller bundling.

This script will:
 - enumerate important files from a Tesseract installation (exe, DLLs, tessdata)
 - copy them into a local `tesseract_bundle/` directory
 - print a recommended pyinstaller command that includes the copied files

Usage:
    python scripts/bundle_tesseract.py --source "C:\\Program Files\\Tesseract-OCR" --dest tesseract_bundle

The script is conservative: it copies tesseract.exe, all DLLs from the install
folder and the entire `tessdata` directory. Use --dry-run to only list actions.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def find_tesseract_install(default: str = r"C:\Program Files\Tesseract-OCR") -> Path:
    """Return an existing Tesseract install path or raise FileNotFoundError.

    :param default: default path to check on Windows
    """
    p = Path(default)
    if p.exists() and p.is_dir():
        return p
    raise FileNotFoundError(f"Tesseract installation not found at {default}")


def list_required_files(install_dir: Path) -> Tuple[Path, List[Path], Path]:
    """Return (exe_path, dll_paths, tessdata_path).

    We consider:
    - tesseract.exe in root
    - all .dll files in root
    - tessdata directory (required)
    """
    exe = install_dir / "tesseract.exe"
    dlls = [p for p in install_dir.iterdir() if p.suffix.lower() == ".dll" and p.is_file()]
    tessdata = install_dir / "tessdata"
    return exe, dlls, tessdata


def copy_tree(src: Path, dst: Path, dry_run: bool = False) -> None:
    """Copy directory tree from src to dst. Creates dst if needed."""
    if dry_run:
        print(f"DRY RUN: copy tree {src} -> {dst}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src: Path, dst_dir: Path, dry_run: bool = False) -> Path:
    """Copy single file to dst_dir and return destination path."""
    dst = dst_dir / src.name
    if dry_run:
        print(f"DRY RUN: copy file {src} -> {dst}")
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def build_pyinstaller_command(bundle_dir: Path, copied_dlls: List[Path],
                              tessdata_relative: str = "tessdata", shell: str = "bash") -> str:
    """Return a recommended pyinstaller command for the given shell.

    :param shell: one of 'bash', 'powershell', 'cmd'
    """
    # Chooses path separator for add-data/add-binary depending on target OS
    sep = ";" if shell in ("powershell", "cmd") else ":"
    parts: List[str] = ["pyinstaller --onefile --windowed"]
    # tesseract.exe
    parts.append(f'--add-binary "{bundle_dir / "tesseract.exe"}{sep}."')
    # DLLs / binaries
    for dll in copied_dlls:
        parts.append(f'--add-binary "{dll}{sep}."')
    # tessdata folder
    parts.append(f'--add-data "{bundle_dir / tessdata_relative}{sep}./tessdata"')
    parts.append("ocr_app.py")

    if shell == "bash":
        return " \\\n  ".join(parts)
    if shell == "powershell":
        # PowerShell uses backtick ` as line continuation
        return " `\n  ".join(parts)
    # cmd / Windows default: single-line (caret ^ is fragile in different contexts)
    return " ".join(parts)


def write_script_file(path: Path, command: str, shell: str, dry_run: bool = False) -> None:
    """Write a bootstrap script containing the pyinstaller command for the chosen shell."""
    if dry_run:
        print(f"DRY RUN: would write script to {path}")
        return
    content_lines: List[str] = []
    if shell == "bash":
        content_lines = ["#!/usr/bin/env bash", "set -e", "", command, ""]
    elif shell == "powershell":
        content_lines = ["# PowerShell build script", "", command, ""]
    else:  # cmd
        content_lines = ["@echo off", "", command, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content_lines), encoding="utf-8")
    # make executable on unix-like systems
    try:
        if shell == "bash" and os.name != "nt":
            st = path.stat().st_mode
            path.chmod(st | 0o111)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle Tesseract files for PyInstaller")
    parser.add_argument("--source", "-s", type=str, default=r"C:\Program Files\Tesseract-OCR", help="Tesseract install folder")
    parser.add_argument("--dest", "-d", type=str, default="tesseract_bundle", help="Destination bundle folder inside project")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    parser.add_argument("--shell", choices=["bash", "powershell", "cmd"], default=None, help="Target shell for generated pyinstaller command")
    parser.add_argument("--write-script", type=str, default=None, help="Write a helper build script to this path")
    args = parser.parse_args()

    try:
        install_dir = find_tesseract_install(args.source)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    exe, dlls, tessdata = list_required_files(install_dir)
    if not exe.exists():
        print(f"tesseract.exe not found in {install_dir}", file=sys.stderr)
        return 3
    if not tessdata.exists() or not tessdata.is_dir():
        print(f"tessdata folder not found in {install_dir}", file=sys.stderr)
        return 4

    dest_dir = Path(args.dest).resolve()
    print(f"Destination bundle directory: {dest_dir}")
    print(f"Source install directory: {install_dir}")

    copied_dlls: List[Path] = []
    # copy executable
    copy_file(exe, dest_dir, dry_run=args.dry_run)

    # copy dlls
    for dll in dlls:
        dst = copy_file(dll, dest_dir, dry_run=args.dry_run)
        copied_dlls.append(dst)

    # copy tessdata tree
    copy_tree(tessdata, dest_dir / "tessdata", dry_run=args.dry_run)

    print("\nFiles copied:")
    print(f" - {dest_dir / 'tesseract.exe'}")
    for d in copied_dlls:
        print(f" - {d}")
    print(f" - {dest_dir / 'tessdata'} (directory)")

    # Choose default shell: bash on posix, powershell/cmd on Windows
    chosen_shell = args.shell
    if chosen_shell is None:
        chosen_shell = "powershell" if os.name == "nt" else "bash"

    cmd = build_pyinstaller_command(dest_dir, copied_dlls, tessdata_relative="tessdata", shell=chosen_shell)
    print(f"\nRecommended PyInstaller command ({chosen_shell}):")
    print(cmd)

    # Optionally write a helper script file
    if args.write_script:
        script_path = Path(args.write_script)
        write_script_file(script_path, cmd, shell=chosen_shell, dry_run=args.dry_run)
        print(f"\nWrote helper script: {script_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
