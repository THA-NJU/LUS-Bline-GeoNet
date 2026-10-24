# Architecture and learning objective

## Pixel and geometry pathways

B-GeoNet takes a one-channel image scaled to [0,1]. Rolling-UNet-S produces pixel logits, eight-channel full-resolution features and a 256-channel semantic representation. An adaptive 16 × 16 pooling grid with learned projection of Cartesian coordinates supplies the geometry decoder's memory.

Sixteen learnable 256-dimensional queries pass through four-head cross-attention, four-head self-attention and a 256 → 512 → 256 feed-forward block. Linear heads predict six geometry variables and one confidence logit per query:

\[
\theta_q=(x_s,y_s,x_e,y_e,w_s,w_e),\qquad o_q\in\mathbb{R}.
\]

Endpoints are ordered vertically. The width interpolates linearly along the segment and is measured perpendicular to its axis. The FP32 rasterizer uses four sigmoid factors for the two endpoint and two side constraints on a 128 × 128 grid. Query masks are combined as

\[
G=1-\prod_q(1-\sigma(o_q)R(\theta_q)).
\]

Full-resolution features are projected from 8 to 32 channels. The pixel probability and resized geometry union form a 34-channel tensor. A residual block with two 3 × 3 convolutions, GroupNorm and GELU, followed by a 1 × 1 output projection, produces a correction to the pixel logits:

\[
Z=Z_{\rm pixel}+C\,F([H,\sigma(Z_{\rm pixel}),G]).
\]

The released configuration uses C = 1. The fusion output projection is initialized to zero. Final segmentation is `sigmoid(Z) > 0.5`.

## Geometry supervision

The original binary mask supervises pixel segmentation. Connected components additionally provide geometric targets: a PCA axis defines endpoints, and a least-squares fit over eight cross-sections estimates endpoint widths. Geometric acceptance checks include size, direction, elongation and fitted-mask IoU ≥ 0.65. An image remains in training even when its components do not provide accepted geometric targets.

Hungarian matching associates predictions with accepted instances. Its cost is twice the mean endpoint L1 distance, plus width L1 distance, plus soft instance Dice loss, minus 0.25 times object confidence. Matched masks are supervised at 64 × 64 resolution. All query confidences are supervised for fully reliable images; otherwise only matched queries receive objectness supervision.

## Loss

For either final or pixel logits, segmentation loss is A times binary cross-entropy plus soft Dice loss. The joint objective is

\[
\mathcal L=\mathcal L_{\rm final}
+B(0.5\mathcal L_{\rm pixel}+0.3\mathcal L_{\rm instance}
+0.2\mathcal L_{\rm geometry}+0.1\mathcal L_{\rm object}).
\]

Geometry loss is endpoint Smooth L1 plus width Smooth L1 plus 0.25 times cosine direction loss. Objectness uses binary cross-entropy with logits. The released configuration uses **A = 0.5, B = 0.25**.

Training uses AdamW, learning rate and weight decay 1e-4, ten warmup epochs, cosine scheduling, batch 32, BF16, seed 42, and gradient clipping at norm 1. The highest image-mean validation Dice selects the checkpoint; patience is ten epochs, with a maximum of 600 epochs. Horizontal image, mask and geometry flips provide augmentation.

## Inference implementation

Channel-dependent circular shifts use gather operations. Coordinate grids are device-local FP32 buffers. The runtime compiles the rasterizer and fusion head, uses channels-last weights, and captures fixed-shape execution in a CUDA Graph. The returned probability map shares a reusable buffer; clone it when retaining results across calls.
