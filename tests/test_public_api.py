"""The curated public API surface of ``leworldmodel_judge``."""

from __future__ import annotations

import pytest

import leworldmodel_judge


def test_version_is_exposed():
    assert leworldmodel_judge.__version__ == "0.2.0"


@pytest.mark.parametrize("name", sorted(leworldmodel_judge.__all__))
def test_every_public_name_is_importable(name):
    assert hasattr(leworldmodel_judge, name)
