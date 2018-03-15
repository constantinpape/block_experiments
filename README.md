# Experiments for Scott's Blocks

Extract statistics over segmentation and synapse detections for 5 (or more) larger cutouts
and several smaller cutouts from random locations.

# Plan

First do proof of concept for block 2, for the following workflow

- Map skeletons to alignment (Stefan)
- Calibrate merge heuristics

- Compute skeleton metrics for all neurons of interest for
-- Multicut segmentation
-- Multicut segmentation assembeled from skeletons
-- Watersheds assembeled from skeletons
-- MWS Segmentation (???)

- Use results for skeleton nodes / edges to guide proofreading
-- False split edges iff we don't use assembeled segmentation
-- Nodes with high distances as indicator for false merges

- Compute skeleton metrics for all skeletons (Need the lists from Scott)

- Synapses:
-- Get count of synaptic clefts per object by intersecting the object mask with connected components of Larissa's detections
-- Get synaptic partner counts by extracting, for each overlaping synapse location, the partner ids
