import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from moe_rca.data import (  # noqa: E402
    CanonicalCase,
    CaseRef,
    CausalWindow,
    build_candidates,
    build_schema_candidates,
    build_visible_candidates,
    contract_violations,
)

class DataContractTest(unittest.TestCase):
    def test_candidate_interfaces_are_no_gt(self):
        self.assertEqual(
            list(inspect.signature(build_schema_candidates).parameters),
            ["metric_columns", "system"],
        )
        self.assertEqual(
            list(inspect.signature(build_visible_candidates).parameters),
            ["metric_entities", "log_entities", "trace_entities"],
        )

        raw, audited = build_schema_candidates(
            ["time", "frontendservice_cpu", "checkoutservice_latency_p90"], "OB"
        )
        self.assertEqual(raw, ("checkoutservice_latency", "frontendservice"))
        self.assertEqual(audited, ("checkoutservice_latency", "frontend"))

    def test_integrated_candidate_and_legality_contract(self):
        ref = CaseRef("case", Path("/tmp/case"), "RE2", "OB", 100)
        metrics = pd.DataFrame(
            {"time": [99, 100, 101], "frontendservice_cpu": [1.0, np.nan, 2.0]}
        )
        logs = pd.DataFrame({"time_s": [100.0], "entity": ["cartservice"]})
        traces = pd.DataFrame({"time_s": [100.5], "entity": ["frontend"]})
        case = CanonicalCase(ref, metrics, logs, traces)
        candidates = build_candidates(case, CausalWindow(100, 101))
        self.assertEqual(candidates.metric_schema_raw, ("frontendservice",))
        self.assertEqual(candidates.metric_schema_audited, ("frontend",))
        self.assertEqual(candidates.telemetry_visible, ("cartservice", "frontend"))
        self.assertEqual(contract_violations(case), ())

        bad_ref = CaseRef("bad", Path("/tmp/bad"), "RE1", "OB", 200)
        bad = CanonicalCase(bad_ref, metrics, None, None)
        self.assertEqual(contract_violations(bad), ("inject_time_outside_metric_range",))


class Step2ContractTest(unittest.TestCase):
    def test_integrated_router_contract(self):
        import torch
        from moe_rca.moe import HeterogeneousMoE, MoEBatch
        B, E, C, T = 2, 3, 2, 4
        values = torch.tensor([[[1., 2., 3., 4.], [0., 0., 0., 0.]], [[4., 3., 2., 1.], [0., 0., 0., 0.]]])
        value_mask = torch.tensor([[[1,1,1,1],[0,0,0,0]], [[1,1,1,1],[0,0,0,0]]], dtype=torch.bool)
        channel_mask = torch.tensor([[1,0],[1,0]], dtype=torch.bool)
        channel_features = torch.zeros(B, C, 10); channel_features[:,0,0] = 1
        channel_entity = torch.tensor([[0,-1],[0,-1]])
        entity_mask = torch.tensor([[1,1,0],[1,1,1]], dtype=torch.bool)
        log_features = torch.zeros(B,E,9); log_features[0,1,0] = 1
        trace_features = torch.zeros(B,E,11); trace_features[1,1,0] = 1
        modality_mask = torch.zeros(B,E,3,dtype=torch.bool)
        modality_mask[0,0,0]=True; modality_mask[0,1,1]=True
        modality_mask[1,0,0]=True; modality_mask[1,1,2]=True
        batch = MoEBatch((("a","b"),("a","b","c")), entity_mask, values, value_mask, channel_mask, channel_features, channel_entity, log_features, trace_features, modality_mask)
        model = HeterogeneousMoE(d_model=16, hidden_dim=24)
        out = model(batch)
        self.assertEqual(tuple(out.scores.shape), (B,E))
        self.assertTrue(torch.equal(out.gate_weights[~modality_mask], torch.zeros_like(out.gate_weights[~modality_mask])))
        any_modality = modality_mask.any(-1)
        self.assertTrue(torch.allclose(out.gate_weights.sum(-1)[any_modality], torch.ones_like(out.gate_weights.sum(-1)[any_modality])))
        self.assertEqual(out.gate_weights[1,2].sum().detach().item(), 0.0)
        self.assertTrue(torch.isneginf(out.scores[0,2]))
        out.scores[entity_mask].sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
