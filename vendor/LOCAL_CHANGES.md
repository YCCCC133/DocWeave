# Runtime source snapshot

These packages were extracted from the application runtime:

- mineru 4.0.10
- mineru-vl-utils 2.0.5
- mlx-vlm 0.7.4

They include application-specific conversion progress, batch execution and PaddleOCR-VL vision activation lookup changes. The snapshot preserves upstream copyright and license files. Two standalone example image paths were replaced with `example.png` for portability.

The extraction excludes bytecode and model weights. External dependencies remain separately installable using requirements.txt.
