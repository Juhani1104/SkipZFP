# SkipZFP

[![CI](https://github.com/Juhani1104/SkipZFP/actions/workflows/ci.yml/badge.svg)](https://github.com/Juhani1104/SkipZFP/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Juhani1104/SkipZFP/branch/main/graph/badge.svg)](https://codecov.io/gh/Juhani1104/SkipZFP)
[![Zarr v3 codec](https://img.shields.io/badge/zarr-v3%20codec-7b3fbf)](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Zarr v3 codec that makes ZFP-compressed arrays queryable: skip the blocks that can't
match, decode only the ones that might.
