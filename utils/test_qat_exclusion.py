"""Tests for the modular ExclusionRule system in qat_utils."""
import torch.nn as nn


def _make_conv(in_ch, out_ch, groups):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups)


# --- Protocol compliance ---

def test_groupconvrule_is_exclusion_rule():
    from utils.qat_utils import GroupConvRule, ExclusionRule
    assert isinstance(GroupConvRule(), ExclusionRule)

def test_namepattternrule_is_exclusion_rule():
    from utils.qat_utils import NamePatternRule, ExclusionRule
    assert isinstance(NamePatternRule([r"stem"]), ExclusionRule)

def test_typerule_is_exclusion_rule():
    from utils.qat_utils import TypeRule, ExclusionRule
    assert isinstance(TypeRule(nn.Linear), ExclusionRule)


# --- GroupConvRule ---

def test_groupconv_excludes_grouped():
    from utils.qat_utils import GroupConvRule
    rule = GroupConvRule()
    assert rule.should_exclude("a.dw", _make_conv(64, 64, 64))
    assert rule.should_exclude("a.gconv", _make_conv(64, 64, 2))

def test_groupconv_keeps_regular():
    from utils.qat_utils import GroupConvRule
    assert not GroupConvRule().should_exclude("a.conv", _make_conv(64, 64, 1))

def test_groupconv_ignores_non_conv():
    from utils.qat_utils import GroupConvRule
    assert not GroupConvRule().should_exclude("a.linear", nn.Linear(4, 4))


# --- DepthwiseConvRule ---

def test_depthwise_excludes_depthwise():
    from utils.qat_utils import DepthwiseConvRule
    assert DepthwiseConvRule().should_exclude("a.dw", _make_conv(64, 64, 64))

def test_depthwise_keeps_group_non_depthwise():
    from utils.qat_utils import DepthwiseConvRule
    assert not DepthwiseConvRule().should_exclude("a.g", _make_conv(64, 64, 2))

def test_depthwise_keeps_regular():
    from utils.qat_utils import DepthwiseConvRule
    assert not DepthwiseConvRule().should_exclude("a.conv", _make_conv(64, 64, 1))


# --- NamePatternRule ---

def test_namepattern_partial_match():
    from utils.qat_utils import NamePatternRule
    rule = NamePatternRule([r"stem\.conv"])
    assert rule.should_exclude("visual.trunk.stem.conv", nn.Linear(4, 4))

def test_namepattern_no_match():
    from utils.qat_utils import NamePatternRule
    rule = NamePatternRule([r"stem\.conv"])
    assert not rule.should_exclude("visual.trunk.stages.0.conv", nn.Linear(4, 4))

def test_namepattern_multiple_patterns_any_match():
    from utils.qat_utils import NamePatternRule
    rule = NamePatternRule([r"stem", r"stages\.3"])
    assert rule.should_exclude("visual.trunk.stages.3.block", nn.Linear(4, 4))
    assert not rule.should_exclude("visual.trunk.stages.0.block", nn.Linear(4, 4))

def test_namepattern_empty_patterns():
    from utils.qat_utils import NamePatternRule
    assert not NamePatternRule([]).should_exclude("anything", nn.Linear(4, 4))


# --- TypeRule ---

def test_typerule_matches_exact():
    from utils.qat_utils import TypeRule
    assert TypeRule(nn.Linear).should_exclude("fc", nn.Linear(4, 4))

def test_typerule_no_match():
    from utils.qat_utils import TypeRule
    assert not TypeRule(nn.Linear).should_exclude("fc", nn.Conv2d(4, 4, 1))

def test_typerule_multiple_types():
    from utils.qat_utils import TypeRule
    rule = TypeRule(nn.Linear, nn.Conv2d)
    assert rule.should_exclude("fc", nn.Linear(4, 4))
    assert rule.should_exclude("cv", nn.Conv2d(4, 4, 1))


# --- QATConfig backward compatibility ---

def test_qatconfig_default_empty_rules():
    from utils.qat_utils import QATConfig
    assert QATConfig().exclusion_rules == []

def test_qatconfig_no_shared_mutable_default():
    from utils.qat_utils import QATConfig, GroupConvRule
    cfg1 = QATConfig()
    cfg2 = QATConfig()
    cfg1.exclusion_rules.append(GroupConvRule())
    assert cfg2.exclusion_rules == []

def test_qatconfig_existing_kwargs_still_work():
    from utils.qat_utils import QATConfig
    cfg = QATConfig(enabled=True, weight_bw=8, act_bw=8, quant_scheme="tf_enhanced", calib_samples=512)
    assert cfg.calib_samples == 512
    assert cfg.exclusion_rules == []


# --- apply_exclusion_rules (mock sim) ---

class _MockSim:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(64, 64, 3, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.Linear(128, 10),
        )
        self.excluded = []

    def exclude_layers_from_quantization(self, modules):
        self.excluded.extend(modules)


def test_apply_noop_empty_rules():
    from utils.qat_utils import apply_exclusion_rules
    sim = _MockSim()
    apply_exclusion_rules(sim, [])
    assert sim.excluded == []

def test_apply_group_conv_rule():
    from utils.qat_utils import apply_exclusion_rules, GroupConvRule
    sim = _MockSim()
    apply_exclusion_rules(sim, [GroupConvRule()])
    assert len(sim.excluded) == 1
    assert isinstance(sim.excluded[0], nn.Conv2d)
    assert sim.excluded[0].groups > 1

def test_apply_name_pattern_rule():
    from utils.qat_utils import apply_exclusion_rules, NamePatternRule
    sim = _MockSim()
    apply_exclusion_rules(sim, [NamePatternRule([r"^1$"])])
    assert len(sim.excluded) == 1

def test_apply_multiple_rules_union():
    from utils.qat_utils import apply_exclusion_rules, GroupConvRule, TypeRule
    sim = _MockSim()
    apply_exclusion_rules(sim, [GroupConvRule(), TypeRule(nn.Linear)])
    assert len(sim.excluded) == 2
