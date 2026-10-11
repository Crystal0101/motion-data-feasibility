# Data verification after the initial development run

The official download Notice states that `filtered_mmwave.zip` was produced by
filtering and aggregation. It is not an untreated single-frame radar archive.
[Official Notice](https://docs.google.com/document/d/1484VK1Lw7JWQ6FHqIEypx0sZ3xzw2MeG7QhRvA5ihoc/edit).

The pinned official loader and the checked current loader both decode float64
values into five columns. Neither defines their binary field names or units.
The paper writes (x,y,z,Doppler,intensity), but the numbers in this archive suggest
that the last two columns may use a different order. This remains unconfirmed.
Do not label either column with a physical unit based on this suggestion alone.

An independent training-only check sampled the first, middle and last available
frame of every training recording: 646 recordings, 1,938 payloads and 64,771 points.
No development or held-out sensor payloads were opened for this check.

- Sampled fourth-column range: 11.20 to 34.80. The initial scale of five clips it
  completely. Four associated dimensions in the saved training features are
  constant, confirming actual representation loss.
- Sampled fifth-column range: about -4.835 to 4.230, with 16 distinct signed values.
  Dividing it by 1,000 substantially attenuates its numerical variation.
- 487 of 646 training recordings have interior frame-number gaps. Thirty existing
  frames cannot be assumed to be thirty consecutive acquisition slots.

The cause of missing frames, actual timestamps, physical units and aggregation
window direction remain unverified. No online causality or zero-latency claim is
supported. The initial experiment is retained in the research archive and is not replaced
with a silently corrected result.

A new temporal reader must preserve original frame indices, observation masks
and gaps. Until timestamps are verified, gaps are index units, not seconds.
Unconfirmed fields should remain anonymous or use a fixed, explicitly specified
unit-agnostic transform. A new development comparison must establish a useful
nonprivate temporal baseline before a noisy head can support a feasibility claim.

## Corrected development study

The separately recorded `protocols/corrected_development.md` implements the
unit-agnostic transform, preserves slots 1–297 and availability masks, and checks
record-level references. The corrected 108 pooled training features have no
constant dimensions; the fourth-column transform is strictly monotone on the
inspected range. All 861 training/development recordings remain represented,
with 47,715 absent index slots over the full 1–297 grid, including boundaries.
This differs from interior-gap counts in the earlier audit. There are no empty
payloads in the read cohort. The mask denotes file presence, including a possible
zero-point payload, rather than claiming a physical observation at every slot.

`review.ipynb` reports actual corrected results; historical results are retained separately in the research archive.
The corrected task changes from windows to recordings. Neither higher scores nor
the successful reader checks resolve physical units, timestamps, missingness
causes or upstream aggregation direction.
