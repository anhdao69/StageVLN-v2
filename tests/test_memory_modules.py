"""CPU checks of the external writer's temporal and frozen-input contracts."""
import unittest

import torch
from torch.nn import functional as F

from qwen_vl.contracts import WriterStep
from qwen_vl.models import InstructionEncoder, MemoryAdapter, MemoryWriter, VisualTokenProjector
from qwen_vl.models.visual_tokens import premerge_to_raster


class MemoryModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.width = 16
        self.writer = MemoryWriter(slots=4, width=16, layers=3, heads=4, ffn_width=32)
        self.encoder = InstructionEncoder(text_width=24, width=16, heads=4, ffn_width=32)
        self.adapter = MemoryAdapter(text_width=24, width=16)

    def inputs(self, batch=2):
        return (torch.randn(batch, 6, 16), torch.randn(batch, 8, 16),
                torch.ones(batch, 6, dtype=torch.bool), torch.ones(batch, 8, dtype=torch.bool))

    def test_writer_step_is_pure_and_keeps_three_independent_levels(self):
        state = self.writer.initial(2, 'cpu')
        original = state.detach().clone()
        step = self.writer(state, *self.inputs())
        self.assertIsInstance(step, WriterStep)
        self.assertEqual(step.state.shape, (2, 4, 16))
        self.assertEqual(len(step.levels), 3)
        self.assertIs(step.state, step.levels[-1])
        self.assertTrue(torch.equal(state, original))
        self.assertNotEqual(state.data_ptr(), step.state.data_ptr())
        blocks = list(self.writer.blocks)
        parameter_ids = [set(map(id, block.parameters())) for block in blocks]
        for i in range(3):
            for j in range(i):
                self.assertTrue(parameter_ids[i].isdisjoint(parameter_ids[j]))
        for block in blocks:
            norms = [module for module in block.modules() if isinstance(module, torch.nn.LayerNorm)]
            self.assertEqual(len(norms), 5)
            self.assertEqual(len({id(norm.weight) for norm in norms}), 5)
            self.assertTrue(torch.equal(block.gate.weight, torch.zeros_like(block.gate.weight)))
            self.assertTrue(torch.equal(block.gate.bias, torch.full_like(block.gate.bias, -2)))
        inputs = self.inputs()
        torch.testing.assert_close(self.writer(state, *inputs).state, self.writer(state, *inputs).state)
        before = step.state.detach().clone()
        self.writer(step.state, *self.inputs())
        self.assertTrue(torch.equal(before, step.state))
        for module in self.writer.modules():
            self.assertFalse(any('cache' in name for name in vars(module)))

    def test_instruction_masks_padding_and_detaches_lexical_table(self):
        table = torch.nn.Embedding(20, 24)
        lexical = table(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        encoded = self.encoder(lexical, mask)
        altered = lexical.detach().clone()
        altered[~mask] = float('nan')
        torch.testing.assert_close(encoded, self.encoder(altered, mask))
        self.assertEqual(encoded.shape, (2, 8, 16))
        encoded.square().mean().backward()
        self.assertIsNone(table.weight.grad)
        self.assertGreater(self.encoder.projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.encoder.queries.grad.abs().sum().item(), 0)
        lexical.square().mean().backward()
        self.assertGreater(table.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            self.encoder(torch.randn(1, 0, 24), torch.ones(1, 0, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.encoder(torch.randn(1, 513, 24), torch.ones(1, 513, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.encoder(torch.randn(1, 3, 24), torch.zeros(1, 3, dtype=torch.bool))

    def test_writer_masks_both_context_modalities(self):
        visual, instruction, vm, im = self.inputs()
        vm[:, 3:] = False
        im[:, 5:] = False
        state = self.writer.initial(2, 'cpu')
        expected = self.writer(state, visual, instruction, vm, im).state
        visual[~vm] = float('nan')
        instruction[~im] = float('nan')
        torch.testing.assert_close(expected, self.writer(state, visual, instruction, vm, im).state)
        with self.assertRaises(ValueError):
            self.writer(state, visual, instruction, vm & False, im & False)

    def test_initial_reader_loss_reaches_all_trainable_producers(self):
        projector = VisualTokenProjector(visual_width=12, width=16)
        visual, vm = projector(torch.randn(16, 12), (1, 4, 4))
        instructions = self.encoder(torch.randn(1, 5, 24), torch.ones(1, 5, dtype=torch.bool))
        state = self.writer.initial(1, 'cpu')
        updated = self.writer(state, visual, instructions, vm, torch.ones(1, 8, dtype=torch.bool))
        self.adapter(updated.state).square().mean().backward()
        for module in (self.writer, self.encoder, self.adapter, projector):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if 'gate_norm' in name:
                    # Zero gate weights intentionally block their inputs on
                    # the first update, but not attention/FFN or gate weights.
                    self.assertEqual(parameter.grad.abs().sum().item(), 0, name)
                else:
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_eight_writes_backpropagate_to_first_observation(self):
        visual, instruction, vm, im = self.inputs(batch=1)
        first_visual = visual.clone().requires_grad_()
        state = self.writer.initial(1, 'cpu')
        state = self.writer(state, first_visual, instruction, vm, im).state
        for _ in range(7):
            state = self.writer(state, visual, instruction, vm, im).state
        self.adapter(state).square().mean().backward()
        self.assertGreater(first_visual.grad.abs().sum().item(), 0)

    def test_explicit_detach_stops_gradient_at_segment_boundary(self):
        visual, instruction, vm, im = self.inputs(batch=1)
        first_visual = visual.clone().requires_grad_()
        state = self.writer(self.writer.initial(1, 'cpu'), first_visual, instruction, vm, im).state.detach()
        for _ in range(7):
            state = self.writer(state, visual, instruction, vm, im).state
        self.adapter(state).square().mean().backward()
        self.assertIsNone(first_visual.grad)

    def test_reset_routes_gradient_without_detaching_continuing_stream(self):
        previous = torch.randn(2, 4, 16, requires_grad=True)
        original = previous.detach().clone()
        reset = self.writer.reset(previous, torch.tensor([True, False]))
        torch.testing.assert_close(reset[0], self.writer.initial(1, 'cpu')[0])
        torch.testing.assert_close(reset[1], previous[1])
        self.assertTrue(torch.equal(previous, original))
        reset.sum().backward()
        torch.testing.assert_close(previous.grad[0], torch.zeros_like(previous.grad[0]))
        torch.testing.assert_close(previous.grad[1], torch.ones_like(previous.grad[1]))
        torch.testing.assert_close(self.writer.initial_slots.grad, torch.ones_like(self.writer.initial_slots))

    def test_fp32_recurrent_state_under_bfloat16_autocast(self):
        projector = VisualTokenProjector(visual_width=12, width=16)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            visual, vm = projector(torch.randn(16, 12), (1, 4, 4))
            instruction = self.encoder(torch.randn(1, 5, 24), torch.ones(1, 5, dtype=torch.bool))
            self.assertEqual(visual.dtype, torch.float32)
            self.assertEqual(instruction.dtype, torch.float32)
            state = self.writer.initial(1, 'cpu')
            for _ in range(8):
                step = self.writer(state, visual, instruction, vm, torch.ones(1, 8, dtype=torch.bool))
                state = step.state
                self.assertEqual(state.dtype, torch.float32)
                self.assertTrue(all(level.dtype == torch.float32 for level in step.levels))
                self.assertTrue(torch.isfinite(state).all())
            output = self.adapter(state)
        output.float().square().mean().backward()
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in self.writer.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.writer.parameters()))

    def test_inverse_merge_block_order_matches_coordinate_fixture(self):
        raster = torch.arange(4 * 6 * 3, dtype=torch.float32).reshape(1, 4, 6, 3)
        block_rows = raster.reshape(1, 2, 2, 3, 2, 3).permute(0, 1, 3, 2, 4, 5).reshape(24, 3)
        torch.testing.assert_close(premerge_to_raster(block_rows, (1, 4, 6), 2), raster)
        for grid in ((2, 4, 6), (1, 3, 6), (1, 4, 4)):
            with self.assertRaises(ValueError):
                premerge_to_raster(block_rows, grid, 2)

    def test_visual_projection_pool_order_and_normalized_coordinates(self):
        projector = VisualTokenProjector(visual_width=4, width=4)
        with torch.no_grad():
            projector.projection.weight.copy_(torch.eye(4))
            projector.projection.bias.zero_()
            projector.visual_type.zero_()
        for height, width in ((4, 6), (16, 32), (2, 64)):
            raster = torch.arange(height * width * 4, dtype=torch.float32).reshape(1, height, width, 4)
            block_rows = raster.reshape(1, height // 2, 2, width // 2, 2, 4).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4)
            actual, mask = projector(block_rows, (1, height, width))
            zeros, _ = projector(torch.zeros_like(block_rows), (1, height, width))
            h2 = min(height, max(1, round(8 * height / max(height, width))))
            w2 = min(width, max(1, round(8 * width / max(height, width))))
            expected = F.adaptive_avg_pool2d(raster.permute(0, 3, 1, 2), (h2, w2)).flatten(2).transpose(1, 2)
            torch.testing.assert_close(actual - zeros, expected, rtol=1e-5, atol=1e-3)
            self.assertEqual(actual.shape, (1, h2 * w2, 4))
            self.assertEqual(mask.shape, (1, h2 * w2))
            self.assertEqual(mask.dtype, torch.bool)
            self.assertTrue(mask.all())
            self.assertLessEqual(h2 * w2, 64)
            u = (torch.arange(w2) + .5) / w2
            v = (torch.arange(h2) + .5) / h2
            vv, uu = torch.meshgrid(v, u, indexing='ij')
            positions = torch.stack(((2 * torch.pi * uu).sin(), (2 * torch.pi * uu).cos(),
                                     (2 * torch.pi * vv).sin(), (2 * torch.pi * vv).cos()), dim=-1).reshape(1, -1, 4)
            torch.testing.assert_close(zeros, positions)


if __name__ == '__main__':
    unittest.main()
