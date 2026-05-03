"""DOC-02 compliance tests — every LightningModule subclass class-level docstring
must contain Paper, Authors, Venue, arXiv link, Gotchas, Reference implementation.

These tests check the **class** docstring (cls.__doc__), NOT module docstrings.
Per REQUIREMENTS.md DOC-02: 'Per-method docstring in each LightningModule subclass'.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# DOC-02 field checker
# ---------------------------------------------------------------------------

def _check_doc02(cls, venue_year: str, arxiv_fragment: str) -> None:
    """Assert class docstring contains all DOC-02 required fields."""
    doc = cls.__doc__
    assert doc is not None, f"{cls.__name__} must have a class docstring"
    assert "Paper:" in doc, f"{cls.__name__}: missing 'Paper:' field"
    assert "Authors:" in doc, f"{cls.__name__}: missing 'Authors:' field"
    assert "Venue:" in doc, f"{cls.__name__}: missing 'Venue:' field"
    assert "arXiv:" in doc, f"{cls.__name__}: missing 'arXiv:' field"
    assert venue_year in doc, (
        f"{cls.__name__}: docstring missing venue/year '{venue_year}'"
    )
    assert arxiv_fragment in doc, (
        f"{cls.__name__}: docstring missing arXiv fragment '{arxiv_fragment}'"
    )
    assert "Gotcha" in doc or "gotcha" in doc.lower(), (
        f"{cls.__name__}: missing 'Gotchas:' section"
    )
    assert "Reference implementation" in doc, (
        f"{cls.__name__}: missing 'Reference implementation:' field"
    )


# ---------------------------------------------------------------------------
# Era 1: Proxy Tasks
# ---------------------------------------------------------------------------

def test_doc02_instance_discrimination():
    from methods.instance_discrimination.module import InstanceDiscriminationModule
    _check_doc02(InstanceDiscriminationModule, "CVPR 2018", "1805.01978")


def test_doc02_invariant_spread():
    from methods.invariant_spread.module import InvariantSpreadModule
    _check_doc02(InvariantSpreadModule, "CVPR 2019", "arxiv.org")


# ---------------------------------------------------------------------------
# Era 2: In-Batch / Queue / Prototype
# ---------------------------------------------------------------------------

def test_doc02_simclr_v1():
    from methods.simclr.module import SimCLRv1Module
    _check_doc02(SimCLRv1Module, "ICML 2020", "2002.05709")


def test_doc02_simclr_v2():
    from methods.simclr.module import SimCLRv2Module
    _check_doc02(SimCLRv2Module, "NeurIPS 2020", "2006.10029")


def test_doc02_moco_v1():
    from methods.moco.module import MoCoV1Module
    _check_doc02(MoCoV1Module, "CVPR 2020", "1911.05722")


def test_doc02_moco_v2():
    from methods.moco.module import MoCoV2Module
    _check_doc02(MoCoV2Module, "2020", "2003.04297")


def test_doc02_swav():
    from methods.swav.module import SwAVModule
    # SwAVModule class docstring was a one-liner placeholder before Plan 10-02 Task 1.
    # Verify the placeholder text is gone.
    assert "see module-level docstring" not in (SwAVModule.__doc__ or "")
    _check_doc02(SwAVModule, "NeurIPS 2020", "2006.09882")


def test_doc02_infomin():
    from methods.infomin.module import InfoMinModule
    _check_doc02(InfoMinModule, "NeurIPS 2020", "2005.10243")


# ---------------------------------------------------------------------------
# Era 3: No-Negative
# ---------------------------------------------------------------------------

def test_doc02_byol():
    from methods.byol.module import BYOLModule
    _check_doc02(BYOLModule, "NeurIPS 2020", "2006.07733")


def test_doc02_simsiam():
    from methods.simsiam.module import SimSiamModule
    _check_doc02(SimSiamModule, "CVPR 2021", "2011.10566")


def test_doc02_barlow_twins():
    from methods.barlow_twins.module import BarlowTwinsModule
    _check_doc02(BarlowTwinsModule, "ICML 2021", "2103.03230")


# ---------------------------------------------------------------------------
# Era 4: Transformer
# ---------------------------------------------------------------------------

def test_doc02_moco_v3():
    from methods.moco_v3.module import MoCoV3Module
    _check_doc02(MoCoV3Module, "ICCV 2021", "2104.02057")


def test_doc02_dino():
    from methods.dino.module import DINOModule
    _check_doc02(DINOModule, "ICCV 2021", "2104.14294")


# ---------------------------------------------------------------------------
# Supervised Contrastive
# ---------------------------------------------------------------------------

def test_doc02_supcon():
    from methods.supcon.module import SupConModule
    _check_doc02(SupConModule, "NeurIPS 2020", "2004.11362")
    # SupCon-specific gotcha that REQUIREMENTS.md SUP-01 calls out:
    assert "ClassBalancedSampler" in SupConModule.__doc__


def test_doc02_supcon_finetune():
    from methods.supcon.module import SupConFinetuneModule
    _check_doc02(SupConFinetuneModule, "NeurIPS 2020", "2004.11362")
    # SupConFinetune-specific gotcha — weight_decay=0.0 on linear head:
    assert "weight_decay=0.0" in SupConFinetuneModule.__doc__


# ---------------------------------------------------------------------------
# Helper unit test
# ---------------------------------------------------------------------------

def test_doc02_helper_rejects_missing_field():
    """_check_doc02 must raise AssertionError when a required field is absent."""
    class _BadDoc:
        """Some method.

        Authors: nobody
        Venue: ICML 2099
        arXiv: https://arxiv.org/abs/0000.00000
        Gotchas:
        - none
        Reference implementation: https://example.com
        """
    # Missing "Paper:" — should fail
    with pytest.raises(AssertionError, match="missing 'Paper:'"):
        _check_doc02(_BadDoc, "ICML 2099", "0000.00000")
