# Pretrained checkpoints

The four `PaleoLandmarker-*-pretrained.pt` files are formal context-and-resize
models pretrained on the six-domain Synthetic dataset at 224x224 resolution.
They contain model weights plus the training epoch, validation error, and
configuration metadata; optimizer state is not included.

The pretrained head has 18 Synthetic landmarks. During transfer learning,
`paleolandmarker.training.checkpoints.load_compatible` loads compatible
encoder tensors and leaves the target-specific prediction head trainable, so
the checkpoints can be used with real datasets having different landmark
counts.
