# V1 Plan — Multi-Frame Spatial Forcing with JanusVLN-Style VGGT Memory

## Goal

Upgrade **v0 single-frame Spatial Forcing**

\[
Q_t \leftrightarrow VGGT(I_t)
\]

to **v1 trajectory-aware Spatial Forcing**

\[
Q_t \leftrightarrow VGGT(I_t \mid H_t)
\]

while keeping the architecture, loss, and inference pipeline otherwise unchanged.

## 1. Keep the QwenVL Branch Unchanged

Use the existing input:

```text
Instruction
+ 8 history frames
+ current frame
→ QwenVL
```

Continue extracting only the **current-frame hidden tokens** from the same Qwen layer as v0.

No changes to:
- navigation input
- action loss
- projector
- inference behavior

## 2. Replace Single-Frame VGGT with JanusVLN-Style Causal Multi-Frame VGGT

Use the **same 8 history frames + current frame** available to Qwen.

Following JanusVLN, process frames **sequentially** through VGGT while carrying the global-attention KV cache:

```text
reset VGGT KV cache

H1      → VGGT              → cache1
H2      → VGGT(cache1)      → cache2
...
H8      → VGGT(cache7)      → cache8
Current → VGGT(cache8)      → G_current
```

Do **not** run the frames as independent single-frame VGGT calls.

The purpose is to obtain a current-frame geometry representation conditioned on the trajectory history:

\[
G_t = VGGT_{stream}(I_t \mid H_t)
\]

## 3. Use Only the Final / Current VGGT Feature

After processing all history frames, keep only the VGGT feature corresponding to the **current frame**:

\[
G_t = VGGT_{stream}(H_1, ..., H_8, I_t)_{current}
\]

Do not align historical Qwen tokens individually.

Then apply the same:
- VGGT layer selection
- spatial resizing
- token alignment

as in v0.

## 4. Keep the Spatial Forcing Loss Unchanged

Use:

\[
L_{SF-v1}
=
1 - \cos(P(Q_t), G_t)
\]

and

\[
L =
L_{nav}
+
\lambda L_{SF-v1}
\]

For the first v1 experiment, keep the same:
- Qwen layer
- VGGT layer
- projector
- \(\lambda\)
- learning rates
- batch size
- training steps
- Qwen history sampler

as v0.

**The only intended change is the VGGT target: single-frame → causal multi-frame.**

## 5. Training Cache Behavior

For every training sample:

```text
VGGT cache = None

H1 → H2 → ... → H8 → Current

extract G_current

discard VGGT cache
```

Do not carry VGGT cache between unrelated training samples.

## 6. Inference Remains Identical to v0

VGGT remains **training-only**:

```text
Instruction + history + current
            ↓
          QwenVL
            ↓
          Action
```

At inference:
- no VGGT
- no VGGT KV cache
- no geometry merger
- no additional geometry encoder

## Minimal v0 → v1 Change

### v0

```text
Current
   ↓
 VGGT
   ↓
G_current
```

### v1

```text
H1 → H2 → ... → H8 → Current
        VGGT + causal KV cache
                 ↓
             G_current
```

Everything after `G_current` stays unchanged.

## First Controlled Experiment

| Model | VGGT Teacher |
|---|---|
| SFT | None |
| v0 | `VGGT(Current)` |
| **v1** | `VGGT_stream(H1...H8, Current)` |

Primary question:

\[
\boxed{
\text{Does trajectory-conditioned VGGT supervision improve over single-frame VGGT supervision?}
}
\]

If v1 clearly improves SR/SPL over v0, proceed to memory-length and frame-sampling ablations.
