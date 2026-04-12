import argparse
import sys

import xs_infer_bathy_raster
from channel_template_invariant import enforce_channel_template_invariant


def test_parse_args_disables_channel_template_by_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['xs_infer_bathy_raster.py'])
    args = xs_infer_bathy_raster._parse_args()
    assert args.channel_template_enabled is False
    enforced = xs_infer_bathy_raster._enforce_channel_template_invariant(args)
    assert enforced.channel_template_enabled is False
    assert enforced.channel_template_requested is False
    assert enforced.channel_template_forced_enabled is False


def test_direct_xs_invariant_preserves_explicit_disable_request():
    args = argparse.Namespace(channel_template_enabled=False)
    enforced = xs_infer_bathy_raster._enforce_channel_template_invariant(args)
    assert enforced.channel_template_enabled is False
    assert enforced.channel_template_requested is False
    assert enforced.channel_template_forced_enabled is False
    assert getattr(enforced, 'channel_template_forced_reason', None) is None


def test_direct_xs_invariant_preserves_enabled_request():
    args = argparse.Namespace(channel_template_enabled=True)
    enforced = xs_infer_bathy_raster._enforce_channel_template_invariant(args)
    assert enforced.channel_template_enabled is True
    assert enforced.channel_template_requested is True
    assert enforced.channel_template_forced_enabled is False
    assert getattr(enforced, "channel_template_forced_reason", None) is None


def test_shared_invariant_helper_normalizes_programmatic_cfg():
    cfg = argparse.Namespace(channel_template_enabled=False)
    requested = enforce_channel_template_invariant(
        cfg,
        enabled_attr="channel_template_enabled",
        requested_attr="channel_template_requested",
        forced_attr="channel_template_forced_enabled",
        reason="test invariant",
    )
    assert requested is False
    assert cfg.channel_template_enabled is False
    assert cfg.channel_template_requested is False
    assert cfg.channel_template_forced_enabled is False


def test_parse_args_no_channel_template_preserves_disable_request(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['xs_infer_bathy_raster.py', '--no-channel-template'])
    args = xs_infer_bathy_raster._enforce_channel_template_invariant(xs_infer_bathy_raster._parse_args())
    assert args.channel_template_enabled is False
    assert args.channel_template_requested is False
    assert args.channel_template_forced_enabled is False
    assert getattr(args, 'channel_template_forced_reason', None) is None


def test_shared_invariant_helper_records_reason_attr():
    cfg = argparse.Namespace(channel_template_enabled=False)
    enforce_channel_template_invariant(
        cfg,
        enabled_attr="channel_template_enabled",
        requested_attr="channel_template_requested",
        forced_attr="channel_template_forced_enabled",
        reason_attr="channel_template_forced_reason",
        reason="test invariant reason",
    )
    assert cfg.channel_template_forced_reason is None
