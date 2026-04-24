# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import pytest

from qai_hub_models.utils.asset_loaders import CachedWebAsset, ModelZooAssetConfig


def _make_asset_config(tmpdir: str) -> ModelZooAssetConfig:
    """Create a minimal ModelZooAssetConfig rooted at tmpdir."""
    return ModelZooAssetConfig(
        asset_url="https://example.com",
        web_asset_folder="",
        static_web_banner_filename="",
        animated_web_banner_filename="",
        model_asset_folder="",
        dataset_asset_folder="",
        local_store_path=tmpdir,
        qaihm_repo="",
        example_use="",
        huggingface_path="",
        repo_url="",
        models_website_url="",
        models_website_relative_path="",
        genie_url="",
        released_asset_folder="",
        released_asset_filename="",
        released_asset_with_chipset_filename="",
    )


def _make_zip(zip_path: Path, files: dict[str, bytes]) -> None:
    """Create a zip file containing the given files."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _make_tar(tar_path: Path, files: dict[str, bytes]) -> None:
    """Create a tar.gz file containing the given files."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


class TestCachedWebAssetInit:
    """Tests for CachedWebAsset.__init__ and is_extracted detection."""

    def test_default_extracted_path_is_archive_stem(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            local_cache = Path(tmpdir) / "data.zip"
            asset = CachedWebAsset(
                "https://example.com/data.zip",
                local_cache,
                cfg,
            )
            assert asset._local_cache_extracted_path == Path(tmpdir) / "data"

    def test_custom_extracted_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/data/train.zip",
                Path(tmpdir) / "data" / "train.zip",
                cfg,
                local_cache_extracted_path=Path(tmpdir) / "data",
            )
            assert asset._local_cache_extracted_path == Path(tmpdir) / "data"

    def test_is_extracted_false_when_nothing_on_disk(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/foo.zip",
                Path(tmpdir) / "foo.zip",
                cfg,
            )
            assert not asset.is_extracted

    def test_is_extracted_true_when_extracted_dir_has_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            extracted = Path(tmpdir) / "foo"
            extracted.mkdir()
            (extracted / "file.txt").write_bytes(b"data")
            asset = CachedWebAsset(
                "https://example.com/foo.zip",
                Path(tmpdir) / "foo.zip",
                cfg,
            )
            assert asset.is_extracted

    def test_is_extracted_false_when_extracted_dir_is_empty(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            (Path(tmpdir) / "foo").mkdir()
            asset = CachedWebAsset(
                "https://example.com/foo.zip",
                Path(tmpdir) / "foo.zip",
                cfg,
            )
            assert not asset.is_extracted

    def test_is_extracted_true_for_custom_path_with_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "train").mkdir()
            asset = CachedWebAsset(
                "https://example.com/data/train.zip",
                Path(tmpdir) / "data" / "train.zip",
                cfg,
                local_cache_extracted_path=data_dir,
            )
            assert asset.is_extracted

    def test_is_extracted_false_for_custom_path_when_dir_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/data/train.zip",
                Path(tmpdir) / "data" / "train.zip",
                cfg,
                local_cache_extracted_path=Path(tmpdir) / "data",
            )
            assert not asset.is_extracted

    def test_is_extracted_raises_for_non_archive(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/model.pt",
                Path(tmpdir) / "model.pt",
                cfg,
            )
            assert not asset.archive_ext
            with pytest.raises(ValueError, match="not an archive"):
                asset.is_extracted  # noqa: B018

    def test_non_archive_extracted_path_is_none(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/model.pt",
                Path(tmpdir) / "model.pt",
                cfg,
            )
            assert asset._local_cache_extracted_path is None

    def test_is_archive_true_for_zip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/data.zip",
                Path(tmpdir) / "data.zip",
                cfg,
            )
            assert asset.archive_ext

    def test_is_archive_false_for_non_archive(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/model.pt",
                Path(tmpdir) / "model.pt",
                cfg,
            )
            assert not asset.archive_ext


class TestCachedWebAssetPath:
    """Tests for CachedWebAsset.path property."""

    def test_path_returns_archive_path_when_not_extracted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            local_cache = Path(tmpdir) / "foo.zip"
            asset = CachedWebAsset(
                "https://example.com/foo.zip",
                local_cache,
                cfg,
            )
            assert asset.path == local_cache

    def test_path_returns_extracted_path_when_extracted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            extracted = Path(tmpdir) / "foo"
            extracted.mkdir()
            (extracted / "file.txt").write_bytes(b"data")
            asset = CachedWebAsset(
                "https://example.com/foo.zip",
                Path(tmpdir) / "foo.zip",
                cfg,
            )
            assert asset.is_extracted
            assert asset.path == extracted

    def test_path_with_custom_extracted_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "train").mkdir()
            asset = CachedWebAsset(
                "https://example.com/data/train.zip",
                Path(tmpdir) / "data" / "train.zip",
                cfg,
                local_cache_extracted_path=data_dir,
            )
            assert asset.path == data_dir

    def test_extracted_path_property(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/foo.zip",
                Path(tmpdir) / "foo.zip",
                cfg,
            )
            assert asset.extracted_path == Path(tmpdir) / "foo"

    def test_extracted_path_raises_for_non_archive(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedWebAsset(
                "https://example.com/model.pt",
                Path(tmpdir) / "model.pt",
                cfg,
            )
            with pytest.raises(ValueError, match="not an archive"):
                asset.extracted_path  # noqa: B018

    def test_path_returns_local_cache_for_non_archive(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            local_cache = Path(tmpdir) / "model.pt"
            asset = CachedWebAsset(
                "https://example.com/model.pt",
                local_cache,
                cfg,
            )
            assert asset.path == local_cache


class TestCachedWebAssetExtract:
    """Tests for CachedWebAsset.extract()."""

    def test_extract_zip_default_path(self) -> None:
        """Zip with flat files extracts to <stem>/ directory."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            zip_path = Path(tmpdir) / "data.zip"
            _make_zip(zip_path, {"file1.txt": b"hello", "file2.txt": b"world"})

            asset = CachedWebAsset(
                "https://example.com/data.zip",
                zip_path,
                cfg,
            )
            result = asset.extract()

            assert result == Path(tmpdir) / "data"
            assert (Path(tmpdir) / "data" / "file1.txt").read_bytes() == b"hello"
            assert (Path(tmpdir) / "data" / "file2.txt").read_bytes() == b"world"
            assert not zip_path.exists()
            assert asset.is_extracted

    def test_extract_zip_custom_path_avoids_double_nesting(self) -> None:
        """Zip with top-level dir extracts to custom parent to avoid double-nesting."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            # Put zip outside the extracted dir to test real extraction
            src_dir = Path(tmpdir) / "_src"
            src_dir.mkdir()
            zip_path = src_dir / "train.zip"
            _make_zip(
                zip_path,
                {"train/img1.jpg": b"img1", "train/img2.jpg": b"img2"},
            )

            data_dir = Path(tmpdir) / "data"
            asset = CachedWebAsset(
                "https://example.com/data/train.zip",
                zip_path,
                cfg,
                local_cache_extracted_path=data_dir,
            )
            result = asset.extract()

            assert result == data_dir
            # Single top-level dir "train" is unwrapped into data_dir
            assert (data_dir / "img1.jpg").read_bytes() == b"img1"
            assert (data_dir / "img2.jpg").read_bytes() == b"img2"
            assert not zip_path.exists()
            assert asset.is_extracted

    def test_extract_tar_default_path(self) -> None:
        """Tar extracts into the extracted_path directory."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            tar_path = Path(tmpdir) / "archive.tgz"
            _make_tar(
                tar_path,
                {"f1.txt": b"one", "f2.txt": b"two"},
            )

            asset = CachedWebAsset(
                "https://example.com/archive.tgz",
                tar_path,
                cfg,
            )
            result = asset.extract()

            assert result == Path(tmpdir) / "archive"
            assert (Path(tmpdir) / "archive" / "f1.txt").read_bytes() == b"one"
            assert (Path(tmpdir) / "archive" / "f2.txt").read_bytes() == b"two"
            assert not tar_path.exists()
            assert asset.is_extracted

    def test_extract_non_archive_raises(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            (Path(tmpdir) / "data.bin").touch()
            asset = CachedWebAsset(
                "https://example.com/data.bin",
                Path(tmpdir) / "data.bin",
                cfg,
            )
            with pytest.raises(ValueError, match="not an archive"):
                asset.extract()

    def test_extract_idempotent(self) -> None:
        """Second extract() call returns immediately when already extracted."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            zip_path = Path(tmpdir) / "data.zip"
            _make_zip(zip_path, {"a.txt": b"data"})

            asset = CachedWebAsset(
                "https://example.com/data.zip",
                zip_path,
                cfg,
            )
            first = asset.extract()
            second = asset.extract()
            assert first == second

    def test_extract_cleans_up_on_failure(self) -> None:
        """If extraction fails, the extracted directory is removed."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            # Create a file that looks like a zip but isn't valid
            bad_zip = Path(tmpdir) / "bad.zip"
            bad_zip.write_bytes(b"not a zip")

            asset = CachedWebAsset(
                "https://example.com/bad.zip",
                bad_zip,
                cfg,
            )
            with pytest.raises(BadZipFile):
                asset.extract()

            # Extracted dir should be cleaned up
            assert not asset.extracted_path.exists()


class TestCachedWebAssetFetch:
    """Tests for CachedWebAsset.fetch()."""

    def test_fetch_downloads_and_returns_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src_file = Path(tmpdir) / "_src" / "model.pt"
            src_file.parent.mkdir()
            src_file.write_bytes(b"model data")
            local_cache = Path(tmpdir) / "model.pt"

            def fake_download(
                _url: str, dst_path: str | Path, *_a: object, **_kw: object
            ) -> str:
                shutil.copy2(src_file, str(dst_path))
                return str(dst_path)

            asset = CachedWebAsset(
                "https://example.com/model.pt",
                local_cache,
                cfg,
                model_downloader=fake_download,
            )
            result = asset.fetch()
            assert result == local_cache
            assert result.read_bytes() == b"model data"

    def test_fetch_skips_download_when_cached(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            local_cache = Path(tmpdir) / "model.pt"
            local_cache.write_bytes(b"cached")

            download_called = False

            def tracking_download(
                _url: str, dst_path: str | Path, *_a: object, **_kw: object
            ) -> str:
                nonlocal download_called
                download_called = True
                return str(dst_path)

            asset = CachedWebAsset(
                "https://example.com/model.pt",
                local_cache,
                cfg,
                model_downloader=tracking_download,
            )
            result = asset.fetch()
            assert result == local_cache
            assert not download_called

    def test_fetch_with_extract(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src_zip = Path(tmpdir) / "_src" / "data.zip"
            _make_zip(src_zip, {"a.txt": b"hello"})
            local_cache = Path(tmpdir) / "data.zip"

            def fake_download(
                _url: str, dst_path: str | Path, *_a: object, **_kw: object
            ) -> str:
                shutil.copy2(src_zip, str(dst_path))
                return str(dst_path)

            asset = CachedWebAsset(
                "https://example.com/data.zip",
                local_cache,
                cfg,
                model_downloader=fake_download,
            )
            result = asset.fetch(extract=True)
            assert result == Path(tmpdir) / "data"
            assert (Path(tmpdir) / "data" / "a.txt").read_bytes() == b"hello"

    def test_fetch_with_local_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            local_file = Path(tmpdir) / "_local" / "data.zip"
            _make_zip(local_file, {"x.txt": b"local"})
            local_cache = Path(tmpdir) / "data.zip"

            asset = CachedWebAsset(
                "https://example.com/data.zip",
                local_cache,
                cfg,
            )
            result = asset.fetch(extract=True, local_path=local_file)
            assert result == Path(tmpdir) / "data"
            assert (Path(tmpdir) / "data" / "x.txt").read_bytes() == b"local"

    def test_fetch_returns_extracted_path_when_already_extracted(self) -> None:
        """If already extracted, fetch returns extracted_path without downloading."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            extracted = Path(tmpdir) / "data"
            extracted.mkdir()
            (extracted / "a.txt").write_bytes(b"existing")

            download_called = False

            def tracking_download(
                _url: str, dst_path: str | Path, *_a: object, **_kw: object
            ) -> str:
                nonlocal download_called
                download_called = True
                return str(dst_path)

            asset = CachedWebAsset(
                "https://example.com/data.zip",
                Path(tmpdir) / "data.zip",
                cfg,
                model_downloader=tracking_download,
            )
            result = asset.fetch(extract=True)
            assert result == extracted
            assert not download_called
