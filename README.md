# Residual Perception Preprocessor

A High-Level Policy (HLP) preprocessing module for the "Sweep to Shapes" bimanual manipulation task. This module segments Lego blocks from camera images, overlays goal shapes, and optimizes the placement to minimize manipulation difficulty.

## Overview

This project implements the preprocessing pipeline for a hierarchical policy structure designed to control bimanual robotic arms in sweeping scattered Lego blocks into letter/number shapes. The preprocessor:

1. **Segments** the Lego pile from the main camera image using SAM3
2. **Smooths** the segmentation mask using morphological operations
3. **Optimizes** the goal shape placement to minimize sweep difficulty
4. **Renders** visualizations showing which areas need to be swept away

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- SAM3 model weights at `sam3/sam3.pt`

### Dependencies

```bash
pip install numpy opencv-python scipy torch torchvision pillow
```

The SAM3 model code is included in the `sam3/` directory.

## Quick Start

### Command Line

```bash
# Basic usage
python demo.py --image input.jpg --goal data/E.png --output result.png

# With custom settings
python demo.py --image input.jpg --goal data/Z.png \
    --optimizer hybrid \
    --render goal+residual \
    --lambda-sweep 2.0 \
    --save-masks
```

### Python API

```python
from hlp_preprocessor import HLPPreprocessor

# Create preprocessor
preprocessor = HLPPreprocessor(optimizer_type='hybrid')

# Process an image
output = preprocessor.process(
    image_path='camera_image.jpg',
    goal_image_path='data/E.png',
    render='goal+residual',
    segmentation_prompt='red Lego blocks'
)

# Access results
print(f"Transform: {preprocessor.transform_params}")
print(f"Cost: {preprocessor.optimization_cost}")
costs = preprocessor.get_cost_breakdown()
```

## Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Main Camera    │────▶│  SAM3 Segment   │────▶│  Mask Smoothing │
│  Image          │     │  "Lego blocks"  │     │  (morphological)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│  Goal Image     │────▶│  Binarize       │              │
│  (E.png, etc.)  │     │  (Otsu thresh)  │              │
└─────────────────┘     └────────┬────────┘              │
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────────────────────┐
                        │  Optimize Transformation T      │
                        │  minimize J(T) = λ₁C_fill +     │
                        │  λ₂C_remove + λ₃C_edge +        │
                        │  λ₄C_sweep                      │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │  Render Output                  │
                        │  • Goal overlay (green)         │
                        │  • Residual overlay (purple)    │
                        └─────────────────────────────────┘
```

## Cost Function

The optimization minimizes:

$$J(T) = \lambda_1 \cdot C_{fill} + \lambda_2 \cdot C_{remove} + \lambda_3 \cdot C_{edge} + \lambda_4 \cdot C_{sweep}$$

| Cost Term | Description | Formula |
|-----------|-------------|---------|
| **C_fill** | Penalizes areas that need filling (goal outside pile) | $\sum_p \mathbb{I}(p \in M_{goal} \land p \notin M_{pile})$ |
| **C_remove** | Penalizes total removal area | $\sum_p \mathbb{I}(p \notin M_{goal} \land p \in M_{pile})$ |
| **C_edge** | Rewards edge alignment (negative cost) | $-\sum_p \exp(-d^2/2\sigma^2)$ |
| **C_sweep** | Penalizes removing deep/interior pixels | $\sum_p \mathbb{I}(remove) \cdot D_{pile}(p)^\alpha$ |

## Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `grid` | Discrete grid search over parameter space | Exhaustive search, guaranteed global |
| `scipy` | L-BFGS-B gradient-based optimization | Fast local optimization |
| `differential_evolution` | Evolutionary global optimization | Complex cost landscapes |
| `hybrid` | Coarse grid + local refinement | **Recommended** - balance of speed and quality |

## Configuration

### Optimization Config

```python
from hlp_preprocessor import OptimizationConfig

config = OptimizationConfig(
    lambda_fill=1.0,        # Weight for fill cost
    lambda_remove=1.0,      # Weight for removal cost
    lambda_edge=0.5,        # Weight for edge alignment
    lambda_sweep=1.0,       # Weight for sweep difficulty
    sigma_edge=10.0,        # Gaussian sigma for edge cost
    alpha_sweep=2.0,        # Exponent for distance field
    grid_resolution=20,     # Grid search resolution
    scale_range=(0.3, 1.5), # Allowed scale factors
    theta_range=(-π/4, π/4) # Allowed rotation angles
)
```

### Smoothing Config

```python
from hlp_preprocessor import SmoothingConfig

config = SmoothingConfig(
    kernel_size=5,          # Morphological kernel size
    gaussian_sigma=2.0,     # Gaussian blur sigma
    morph_iterations=2,     # Number of morph operations
    use_closing=True,       # Fill small holes
    use_opening=True        # Remove small noise
)
```

### Render Config

```python
render_config = {
    'color_goal': (0, 255, 0),      # Green for goal shape
    'color_residual': (255, 0, 255), # Magenta for areas to sweep
    'alpha': 0.5                     # Overlay transparency
}
```

## Render Modes

| Mode | Description |
|------|-------------|
| `None` | Original image, no overlay |
| `goal` | Shows goal shape overlay (green) |
| `residual` | Shows areas to sweep away (purple) |
| `goal+residual` | Shows both overlays |

## API Reference

### HLPPreprocessor

Main class that orchestrates the preprocessing pipeline.

```python
class HLPPreprocessor:
    def __init__(self,
                 segmenter=None,           # LegoSegmenter instance
                 smoother=None,            # MaskSmoother instance
                 optimization_config=None, # OptimizationConfig
                 optimizer_type='hybrid',  # Optimizer type
                 render_config=None):      # Render settings
        ...

    def process(self,
                image_path: str,
                goal_image_path: str,
                render: str = None,
                segmentation_prompt: str = "red Lego blocks") -> np.ndarray:
        """Run full preprocessing pipeline."""
        ...

    def process_image(self,
                      image: np.ndarray,
                      goal_image_path: str,
                      render: str = None,
                      segmentation_prompt: str = "red Lego blocks") -> np.ndarray:
        """Process with image array instead of path."""
        ...

    def get_cost_breakdown(self) -> dict:
        """Get detailed cost breakdown after processing."""
        ...
```

### LegoSegmenter

Segments Lego blocks using SAM3.

```python
class LegoSegmenter:
    def __init__(self,
                 model_path: str = "sam3/sam3.pt",
                 device: str = "cuda",
                 confidence_threshold: float = 0.3):
        ...

    def segment(self,
                image: np.ndarray,
                prompt: str = "red Lego blocks") -> np.ndarray:
        """Returns binary mask (H, W)."""
        ...
```

### GoalImageProcessor

Processes and transforms goal images.

```python
class GoalImageProcessor:
    @staticmethod
    def load_and_binarize(image_path: str,
                          invert: bool = None,
                          threshold: int = None) -> np.ndarray:
        """Load goal image and convert to binary mask."""
        ...

    @staticmethod
    def transform_mask(mask: np.ndarray,
                       params: TransformParams,
                       output_shape: tuple) -> np.ndarray:
        """Apply transformation (translate, rotate, scale) to mask."""
        ...
```

### TransformParams

Transformation parameters dataclass.

```python
@dataclass
class TransformParams:
    tx: float = 0.0      # Translation X (pixels)
    ty: float = 0.0      # Translation Y (pixels)
    theta: float = 0.0   # Rotation (radians)
    scale: float = 1.0   # Scale factor
```

## Examples

### Process with Different Goal Shapes

```python
from hlp_preprocessor import HLPPreprocessor

preprocessor = HLPPreprocessor()

for letter in ['E', 'Z', 'N', 'X']:
    output = preprocessor.process(
        image_path='lego_pile.jpg',
        goal_image_path=f'data/{letter}.png',
        render='goal+residual'
    )
    cv2.imwrite(f'result_{letter}.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
```

### Custom Optimizer

```python
from hlp_preprocessor import HLPPreprocessor, OptimizationConfig

# Prioritize minimizing sweep difficulty
config = OptimizationConfig(
    lambda_fill=1.0,
    lambda_remove=0.5,
    lambda_edge=0.3,
    lambda_sweep=2.0,  # Higher weight for sweep difficulty
    alpha_sweep=3.0    # Stronger penalty for deep pixels
)

preprocessor = HLPPreprocessor(
    optimization_config=config,
    optimizer_type='grid'  # Use grid search for best result
)
```

### Access Intermediate Results

```python
preprocessor = HLPPreprocessor()
output = preprocessor.process('image.jpg', 'data/E.png', render='goal+residual')

# Access intermediate masks
lego_mask = preprocessor.mask_lego_smoothed      # Smoothed Lego pile mask
goal_mask = preprocessor.mask_goal_transformed   # Transformed goal mask
residual = preprocessor.mask_residual            # Areas to sweep (pile - goal)

# Access transformation
params = preprocessor.transform_params
print(f"Translation: ({params.tx}, {params.ty})")
print(f"Rotation: {np.degrees(params.theta)}°")
print(f"Scale: {params.scale}")
```

## File Structure

```
Residual-Perception-Preprocessor/
├── hlp_preprocessor.py      # Main module
├── demo.py                  # Command-line demo
├── test_hlp_preprocessor.py # Test suite
├── doc.md                   # Original specification
├── README.md                # This file
├── data/
│   ├── E.png               # Goal shape: letter E
│   ├── Z.png               # Goal shape: letter Z
│   ├── N.png               # Goal shape: letter N
│   └── X.png               # Goal shape: letter X
└── sam3/                   # SAM3 model code
    ├── sam3.pt             # Model weights
    └── sam3/               # SAM3 package
```

## Running Tests

```bash
python test_hlp_preprocessor.py
```

## Command Line Options

```
usage: demo.py [-h] --image IMAGE --goal GOAL [--output OUTPUT]
               [--render {none,goal,residual,goal+residual}]
               [--optimizer {grid,scipy,differential_evolution,hybrid}]
               [--grid-resolution GRID_RESOLUTION]
               [--lambda-fill LAMBDA_FILL] [--lambda-remove LAMBDA_REMOVE]
               [--lambda-edge LAMBDA_EDGE] [--lambda-sweep LAMBDA_SWEEP]
               [--sigma-edge SIGMA_EDGE] [--alpha-sweep ALPHA_SWEEP]
               [--scale-min SCALE_MIN] [--scale-max SCALE_MAX]
               [--color-goal COLOR_GOAL] [--color-residual COLOR_RESIDUAL]
               [--alpha ALPHA] [--smooth-kernel SMOOTH_KERNEL]
               [--smooth-sigma SMOOTH_SIGMA] [--prompt PROMPT]
               [--confidence CONFIDENCE] [--device {cuda,cpu}]
               [--save-masks] [--verbose]
```

## License

This project is part of research on bimanual manipulation for the "Sweep to Shapes" task.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{residual-perception-preprocessor,
  title={Residual Perception Preprocessor for Sweep to Shapes},
  year={2024},
  note={High-Level Policy preprocessing for bimanual manipulation}
}
```
