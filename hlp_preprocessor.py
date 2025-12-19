"""
HLP Preprocessor Module for Residual Perception

This module implements the High Level Policy (HLP) preprocessing pipeline for
the "Sweep to Shapes" manipulation task. It segments Lego blocks using SAM3,
overlays goal shapes, and optimizes the transformation to minimize manipulation difficulty.

Author: Generated for RSS paper
"""

import sys
import os

# Add sam3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam3'))

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Literal, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import ndimage
from scipy.optimize import minimize, differential_evolution
import torch


@dataclass
class TransformParams:
    """Transformation parameters for goal image alignment."""
    tx: float = 0.0  # Translation x (in pixels)
    ty: float = 0.0  # Translation y (in pixels)
    theta: float = 0.0  # Rotation angle (in radians)
    scale: float = 1.0  # Scale factor

    def to_array(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.theta, self.scale])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TransformParams':
        return cls(tx=arr[0], ty=arr[1], theta=arr[2], scale=arr[3])


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    # Cost weights (lambda values)
    lambda_fill: float = 1.0       # λ1: Fill cost weight
    lambda_remove: float = 1.0     # λ2: Remove cost weight
    lambda_edge: float = 0.5       # λ3: Edge alignment weight
    lambda_sweep: float = 1.0      # λ4: Sweepability cost weight

    # Edge alignment parameters
    sigma_edge: float = 10.0       # σ for edge cost Gaussian kernel

    # Sweep difficulty parameters
    alpha_sweep: float = 2.0       # α exponent for distance field

    # Optimizer-specific parameters
    grid_resolution: int = 20      # Resolution for grid search
    scale_range: Tuple[float, float] = (0.3, 1.5)
    theta_range: Tuple[float, float] = (-np.pi/4, np.pi/4)


@dataclass
class SmoothingConfig:
    """Configuration for mask smoothing operations."""
    kernel_size: int = 5           # Morphological kernel size
    gaussian_sigma: float = 2.0    # Gaussian blur sigma
    morph_iterations: int = 2      # Morphological operation iterations
    use_closing: bool = True       # Apply morphological closing
    use_opening: bool = True       # Apply morphological opening


class LegoSegmenter:
    """Segments Lego blocks from images using SAM3."""

    def __init__(self,
                 model_path: str = "/home/whs/manipulation/Residual-Perception-Preprocessor/sam3/sam3.pt",
                 device: str = "cuda",
                 confidence_threshold: float = 0.3):
        """
        Initialize the SAM3-based Lego segmenter.

        Args:
            model_path: Path to SAM3 model weights
            device: Device to run inference on
            confidence_threshold: Confidence threshold for segmentation
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None
        self.processor = None

    def _load_model(self):
        """Lazy load the SAM3 model."""
        if self.model is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.model = build_sam3_image_model(
                checkpoint_path=self.model_path,
                device=self.device,
                eval_mode=True
            )
            self.processor = Sam3Processor(
                self.model,
                device=self.device,
                confidence_threshold=self.confidence_threshold
            )

    def segment(self, image: Union[np.ndarray, Image.Image],
                prompt: str = "red Lego blocks") -> np.ndarray:
        """
        Segment Lego blocks from the input image.

        Args:
            image: Input image (RGB)
            prompt: Text prompt for segmentation

        Returns:
            Binary mask of segmented Lego blocks (H, W), dtype=uint8
        """
        self._load_model()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Set image and get initial state
        state = self.processor.set_image(image)

        # Set text prompt and run inference
        state = self.processor.set_text_prompt(state=state, prompt=prompt)

        # Combine all detected masks into a single binary mask
        if "masks" in state and len(state["masks"]) > 0:
            # Combine all masks using logical OR
            combined_mask = torch.zeros_like(state["masks"][0])
            for mask in state["masks"]:
                combined_mask = combined_mask | mask

            # Convert to numpy and squeeze
            mask_np = combined_mask.squeeze().cpu().numpy().astype(np.uint8)
        else:
            # Return empty mask if no objects detected
            w, h = image.size
            mask_np = np.zeros((h, w), dtype=np.uint8)

        return mask_np


class MaskSmoother:
    """Applies smoothing operations to binary masks."""

    def __init__(self, config: Optional[SmoothingConfig] = None):
        """
        Initialize the mask smoother.

        Args:
            config: Smoothing configuration parameters
        """
        self.config = config or SmoothingConfig()

    def smooth(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply smoothing operations to the input mask.

        Args:
            mask: Binary mask (H, W), dtype=uint8

        Returns:
            Smoothed binary mask (H, W), dtype=uint8
        """
        result = mask.copy()

        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.kernel_size, self.config.kernel_size)
        )

        # Apply morphological closing (fill small holes)
        if self.config.use_closing:
            result = cv2.morphologyEx(
                result, cv2.MORPH_CLOSE, kernel,
                iterations=self.config.morph_iterations
            )

        # Apply morphological opening (remove small noise)
        if self.config.use_opening:
            result = cv2.morphologyEx(
                result, cv2.MORPH_OPEN, kernel,
                iterations=self.config.morph_iterations
            )

        # Apply Gaussian blur and re-threshold
        if self.config.gaussian_sigma > 0:
            result = cv2.GaussianBlur(
                result.astype(np.float32),
                (0, 0),
                self.config.gaussian_sigma
            )
            result = (result > 0.5).astype(np.uint8)

        return result


class GoalImageProcessor:
    """Processes goal images into binary masks."""

    @staticmethod
    def load_and_binarize(image_path: str, invert: Optional[bool] = None,
                          threshold: Optional[int] = None) -> np.ndarray:
        """
        Load a goal image and convert to binary mask.

        Args:
            image_path: Path to the goal image
            invert: If True, dark regions become 1 (foreground).
                   If None, auto-detect based on corner vs center intensity.
            threshold: Manual threshold value. If None, use Otsu's method.

        Returns:
            Binary mask (H, W), dtype=uint8, values 0 or 1
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Use Otsu's method if no threshold specified
        if threshold is None:
            _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

        # Auto-detect inversion if not specified
        if invert is None:
            # Check if corners are brighter than center (white background)
            h, w = img.shape
            corner_mean = np.mean([img[0, 0], img[0, w-1], img[h-1, 0], img[h-1, w-1]])
            center_region = img[h//4:3*h//4, w//4:3*w//4]
            center_mean = np.mean(center_region)
            # If corners are brighter, letters are dark -> invert
            invert = corner_mean > center_mean

        if invert:
            binary = 1 - binary

        return binary.astype(np.uint8)

    @staticmethod
    def transform_mask(mask: np.ndarray,
                       params: TransformParams,
                       output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Apply transformation to a binary mask.

        Args:
            mask: Input binary mask
            params: Transformation parameters
            output_shape: Output shape (H, W)

        Returns:
            Transformed binary mask with output_shape
        """
        h_out, w_out = output_shape
        h_in, w_in = mask.shape

        # Scale the mask first
        if params.scale != 1.0:
            new_h = int(h_in * params.scale)
            new_w = int(w_in * params.scale)
            scaled = cv2.resize(mask.astype(np.float32), (new_w, new_h),
                              interpolation=cv2.INTER_LINEAR)
        else:
            scaled = mask.astype(np.float32)
            new_h, new_w = h_in, w_in

        # Create output canvas
        result = np.zeros((h_out, w_out), dtype=np.float32)

        # Calculate center of scaled mask
        cx_scaled = new_w / 2
        cy_scaled = new_h / 2

        # Calculate center position in output
        cx_out = w_out / 2 + params.tx
        cy_out = h_out / 2 + params.ty

        # Create rotation matrix around the center
        M = cv2.getRotationMatrix2D((cx_scaled, cy_scaled), np.degrees(params.theta), 1.0)

        # Rotate the scaled mask
        rotated = cv2.warpAffine(scaled, M, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Calculate placement coordinates
        x_start = int(cx_out - cx_scaled)
        y_start = int(cy_out - cy_scaled)

        # Calculate valid regions for copying
        src_x_start = max(0, -x_start)
        src_y_start = max(0, -y_start)
        src_x_end = min(new_w, w_out - x_start)
        src_y_end = min(new_h, h_out - y_start)

        dst_x_start = max(0, x_start)
        dst_y_start = max(0, y_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)

        # Copy the valid region
        if src_x_end > src_x_start and src_y_end > src_y_start:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                rotated[src_y_start:src_y_end, src_x_start:src_x_end]

        return (result > 0.5).astype(np.uint8)


class CostFunction:
    """Computes the optimization cost for goal placement."""

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the cost function.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self._pile_mask = None
        self._distance_field = None
        self._pile_edges = None

    def set_pile_mask(self, mask: np.ndarray):
        """
        Set the pile mask and precompute derived quantities.

        Args:
            mask: Binary mask of the Lego pile (H, W)
        """
        self._pile_mask = mask

        # Compute distance field (distance to nearest boundary)
        self._distance_field = ndimage.distance_transform_edt(mask)

        # Compute edge pixels
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        self._pile_edges = dilated - eroded

    def compute_fill_cost(self, goal_mask: np.ndarray) -> float:
        """
        Compute fill cost: pixels in goal but not in pile.

        C_fill = Σ I(p ∈ M_goal ∧ p ∉ M_pile)
        """
        fill_region = goal_mask & (~self._pile_mask.astype(bool))
        return np.sum(fill_region)

    def compute_remove_cost(self, goal_mask: np.ndarray) -> float:
        """
        Compute remove cost: pixels in pile but not in goal.

        C_remove = Σ I(p ∉ M_goal ∧ p ∈ M_pile)
        """
        remove_region = (~goal_mask.astype(bool)) & self._pile_mask.astype(bool)
        return np.sum(remove_region)

    def compute_edge_cost(self, goal_mask: np.ndarray) -> float:
        """
        Compute edge alignment cost (negative = reward).

        C_edge = -Σ exp(-min_dist²/(2σ²))
        """
        # Get goal edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(goal_mask, kernel, iterations=1)
        eroded = cv2.erode(goal_mask, kernel, iterations=1)
        goal_edges = dilated - eroded

        # Find goal edge points
        goal_edge_points = np.argwhere(goal_edges > 0)

        if len(goal_edge_points) == 0:
            return 0.0

        # Find pile edge points
        pile_edge_points = np.argwhere(self._pile_edges > 0)

        if len(pile_edge_points) == 0:
            return 0.0

        # Compute minimum distances using distance transform
        # Create distance field from pile edges
        pile_edge_mask = self._pile_edges > 0
        dist_from_pile_edge = ndimage.distance_transform_edt(~pile_edge_mask)

        # Get distances for goal edge points
        distances = dist_from_pile_edge[goal_edge_points[:, 0], goal_edge_points[:, 1]]

        # Compute Gaussian-weighted reward
        sigma = self.config.sigma_edge
        rewards = np.exp(-distances**2 / (2 * sigma**2))

        return -np.sum(rewards)

    def compute_sweep_cost(self, goal_mask: np.ndarray) -> float:
        """
        Compute sweep difficulty cost.

        C_sweep = Σ [I(p ∉ M_goal ∧ p ∈ M_pile) · D_pile(p)^α]
        """
        # Find pixels to remove
        remove_region = (~goal_mask.astype(bool)) & self._pile_mask.astype(bool)

        # Weight by distance field
        alpha = self.config.alpha_sweep
        weighted_cost = np.sum(
            remove_region * (self._distance_field ** alpha)
        )

        return weighted_cost

    def compute_total_cost(self, goal_mask: np.ndarray) -> float:
        """
        Compute the total optimization cost.

        J(T) = λ1·C_fill + λ2·C_remove + λ3·C_edge + λ4·C_sweep
        """
        c_fill = self.compute_fill_cost(goal_mask)
        c_remove = self.compute_remove_cost(goal_mask)
        c_edge = self.compute_edge_cost(goal_mask)
        c_sweep = self.compute_sweep_cost(goal_mask)

        total = (
            self.config.lambda_fill * c_fill +
            self.config.lambda_remove * c_remove +
            self.config.lambda_edge * c_edge +
            self.config.lambda_sweep * c_sweep
        )

        return total

    def compute_cost_breakdown(self, goal_mask: np.ndarray) -> Dict[str, float]:
        """Get detailed cost breakdown."""
        c_fill = self.compute_fill_cost(goal_mask)
        c_remove = self.compute_remove_cost(goal_mask)
        c_edge = self.compute_edge_cost(goal_mask)
        c_sweep = self.compute_sweep_cost(goal_mask)

        return {
            'fill': c_fill,
            'remove': c_remove,
            'edge': c_edge,
            'sweep': c_sweep,
            'total': (
                self.config.lambda_fill * c_fill +
                self.config.lambda_remove * c_remove +
                self.config.lambda_edge * c_edge +
                self.config.lambda_sweep * c_sweep
            )
        }


class BaseOptimizer(ABC):
    """Abstract base class for transformation optimizers."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cost_fn = CostFunction(config)

    @abstractmethod
    def optimize(self,
                 pile_mask: np.ndarray,
                 goal_mask: np.ndarray) -> Tuple[TransformParams, float]:
        """
        Find optimal transformation parameters.

        Args:
            pile_mask: Binary mask of Lego pile
            goal_mask: Binary mask of goal shape

        Returns:
            Tuple of (optimal parameters, final cost)
        """
        pass

    def _evaluate(self,
                  params: TransformParams,
                  pile_mask: np.ndarray,
                  goal_mask: np.ndarray) -> float:
        """Evaluate cost for given parameters."""
        transformed = GoalImageProcessor.transform_mask(
            goal_mask, params, pile_mask.shape
        )
        return self.cost_fn.compute_total_cost(transformed)


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer for discrete parameter exploration."""

    def optimize(self,
                 pile_mask: np.ndarray,
                 goal_mask: np.ndarray) -> Tuple[TransformParams, float]:
        """
        Find optimal transformation using grid search.
        """
        self.cost_fn.set_pile_mask(pile_mask)

        h, w = pile_mask.shape
        res = self.config.grid_resolution

        # Define search ranges
        tx_range = np.linspace(-w/3, w/3, res)
        ty_range = np.linspace(-h/3, h/3, res)
        theta_range = np.linspace(
            self.config.theta_range[0],
            self.config.theta_range[1],
            max(5, res // 4)
        )
        scale_range = np.linspace(
            self.config.scale_range[0],
            self.config.scale_range[1],
            max(5, res // 4)
        )

        best_cost = float('inf')
        best_params = TransformParams()

        # Grid search
        for scale in scale_range:
            for theta in theta_range:
                for tx in tx_range:
                    for ty in ty_range:
                        params = TransformParams(tx=tx, ty=ty, theta=theta, scale=scale)
                        cost = self._evaluate(params, pile_mask, goal_mask)

                        if cost < best_cost:
                            best_cost = cost
                            best_params = params

        return best_params, best_cost


class ScipyOptimizer(BaseOptimizer):
    """Scipy-based continuous optimizer."""

    def __init__(self, config: OptimizationConfig, method: str = 'L-BFGS-B'):
        """
        Initialize Scipy optimizer.

        Args:
            config: Optimization configuration
            method: Scipy optimization method
        """
        super().__init__(config)
        self.method = method

    def optimize(self,
                 pile_mask: np.ndarray,
                 goal_mask: np.ndarray) -> Tuple[TransformParams, float]:
        """
        Find optimal transformation using Scipy optimization.
        """
        self.cost_fn.set_pile_mask(pile_mask)

        h, w = pile_mask.shape

        def objective(x):
            params = TransformParams.from_array(x)
            return self._evaluate(params, pile_mask, goal_mask)

        # Define bounds
        bounds = [
            (-w/2, w/2),      # tx
            (-h/2, h/2),      # ty
            self.config.theta_range,  # theta
            self.config.scale_range   # scale
        ]

        # Initial guess (centered, no rotation, scale 1)
        x0 = np.array([0.0, 0.0, 0.0, 0.8])

        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds,
            # TODO:
            options={'maxiter': 5000}
        )

        best_params = TransformParams.from_array(result.x)
        return best_params, result.fun


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """Differential evolution optimizer for global optimization."""

    def optimize(self,
                 pile_mask: np.ndarray,
                 goal_mask: np.ndarray) -> Tuple[TransformParams, float]:
        """
        Find optimal transformation using differential evolution.
        """
        self.cost_fn.set_pile_mask(pile_mask)

        h, w = pile_mask.shape

        def objective(x):
            params = TransformParams.from_array(x)
            return self._evaluate(params, pile_mask, goal_mask)

        # Define bounds
        bounds = [
            (-w/2, w/2),      # tx
            (-h/2, h/2),      # ty
            self.config.theta_range,  # theta
            self.config.scale_range   # scale
        ]

        result = differential_evolution(
            objective,
            bounds,
            # TODO:
            maxiter=1000,
            seed=42,
            workers=1,
            updating='deferred'
        )

        best_params = TransformParams.from_array(result.x)
        return best_params, result.fun


class HybridOptimizer(BaseOptimizer):
    """Hybrid optimizer: coarse grid search + local refinement."""

    def optimize(self,
                 pile_mask: np.ndarray,
                 goal_mask: np.ndarray) -> Tuple[TransformParams, float]:
        """
        Find optimal transformation using hybrid approach.
        """
        # First: coarse grid search
        coarse_config = OptimizationConfig(
            lambda_fill=self.config.lambda_fill,
            lambda_remove=self.config.lambda_remove,
            lambda_edge=self.config.lambda_edge,
            lambda_sweep=self.config.lambda_sweep,
            sigma_edge=self.config.sigma_edge,
            alpha_sweep=self.config.alpha_sweep,
            grid_resolution=10,  # Coarse grid
            scale_range=self.config.scale_range,
            theta_range=self.config.theta_range
        )

        grid_optimizer = GridSearchOptimizer(coarse_config)
        coarse_params, _ = grid_optimizer.optimize(pile_mask, goal_mask)

        # Second: local refinement
        self.cost_fn.set_pile_mask(pile_mask)

        def objective(x):
            params = TransformParams.from_array(x)
            return self._evaluate(params, pile_mask, goal_mask)

        h, w = pile_mask.shape

        # Narrow bounds around coarse solution
        bounds = [
            (coarse_params.tx - w/10, coarse_params.tx + w/10),
            (coarse_params.ty - h/10, coarse_params.ty + h/10),
            (coarse_params.theta - np.pi/8, coarse_params.theta + np.pi/8),
            (max(self.config.scale_range[0], coarse_params.scale * 0.8),
             min(self.config.scale_range[1], coarse_params.scale * 1.2))
        ]

        result = minimize(
            objective,
            coarse_params.to_array(),
            method='L-BFGS-B',
            bounds=bounds,
            # TODO:
            options={'maxiter': 2000}
        )

        best_params = TransformParams.from_array(result.x)
        return best_params, result.fun


def create_optimizer(optimizer_type: str, config: OptimizationConfig) -> BaseOptimizer:
    """
    Factory function to create optimizers.

    Args:
        optimizer_type: One of 'grid', 'scipy', 'differential_evolution', 'hybrid'
        config: Optimization configuration

    Returns:
        Optimizer instance
    """
    optimizers = {
        'grid': GridSearchOptimizer,
        'scipy': ScipyOptimizer,
        'differential_evolution': DifferentialEvolutionOptimizer,
        'hybrid': HybridOptimizer
    }

    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. "
                        f"Available: {list(optimizers.keys())}")

    return optimizers[optimizer_type](config)


class Renderer:
    """Renders visualization of masks overlaid on images."""

    def __init__(self,
                 color_goal: Tuple[int, int, int] = (0, 255, 0),      # Bright green
                 color_residual: Tuple[int, int, int] = (255, 0, 255), # Bright purple
                 alpha: float = 0.5):
        """
        Initialize renderer.

        Args:
            color_goal: RGB color for goal mask overlay
            color_residual: RGB color for residual mask overlay
            alpha: Transparency (0-1)
        """
        self.color_goal = color_goal
        self.color_residual = color_residual
        self.alpha = alpha

    def render(self,
               image: np.ndarray,
               mask_goal: Optional[np.ndarray] = None,
               mask_residual: Optional[np.ndarray] = None,
               mode: Optional[Literal['goal', 'residual', 'goal+residual']] = None
               ) -> np.ndarray:
        """
        Render masks overlaid on image.

        Args:
            image: Input image (H, W, 3), RGB
            mask_goal: Goal mask (H, W)
            mask_residual: Residual mask (H, W)
            mode: Render mode - None, 'goal', 'residual', or 'goal+residual'

        Returns:
            Rendered image (H, W, 3), RGB
        """
        if mode is None:
            return image.copy()

        result = image.copy().astype(np.float32)

        if mode in ['goal', 'goal+residual'] and mask_goal is not None:
            # Create goal overlay
            overlay = np.zeros_like(result)
            overlay[mask_goal > 0] = self.color_goal
            mask_3d = np.stack([mask_goal > 0] * 3, axis=-1)
            result = np.where(
                mask_3d,
                result * (1 - self.alpha) + overlay * self.alpha,
                result
            )

        if mode in ['residual', 'goal+residual'] and mask_residual is not None:
            # Create residual overlay
            overlay = np.zeros_like(result)
            overlay[mask_residual > 0] = self.color_residual
            mask_3d = np.stack([mask_residual > 0] * 3, axis=-1)
            result = np.where(
                mask_3d,
                result * (1 - self.alpha) + overlay * self.alpha,
                result
            )

        return result.astype(np.uint8)


class HLPPreprocessor:
    """
    Main HLP Preprocessor class that orchestrates the entire pipeline.

    Pipeline:
    1. Segment Lego blocks using SAM3
    2. Smooth the segmentation mask
    3. Load and binarize goal image
    4. Optimize transformation to align goal with pile
    5. Render output visualization
    """

    def __init__(self,
                 segmenter: Optional[LegoSegmenter] = None,
                 smoother: Optional[MaskSmoother] = None,
                 optimization_config: Optional[OptimizationConfig] = None,
                 optimizer_type: str = 'hybrid',
                 render_config: Optional[Dict] = None):
        """
        Initialize the HLP Preprocessor.

        Args:
            segmenter: LegoSegmenter instance (created if None)
            smoother: MaskSmoother instance (created if None)
            optimization_config: Optimization parameters
            optimizer_type: Type of optimizer to use
            render_config: Rendering configuration
        """
        self.segmenter = segmenter or LegoSegmenter()
        self.smoother = smoother or MaskSmoother()
        self.opt_config = optimization_config or OptimizationConfig()
        self.optimizer = create_optimizer(optimizer_type, self.opt_config)

        render_config = render_config or {}
        self.renderer = Renderer(
            color_goal=render_config.get('color_goal', (0, 255, 0)),
            color_residual=render_config.get('color_residual', (255, 0, 255)),
            alpha=render_config.get('alpha', 0.5)
        )

        # Store intermediate results
        self.mask_lego: Optional[np.ndarray] = None
        self.mask_lego_smoothed: Optional[np.ndarray] = None
        self.mask_goal_binary: Optional[np.ndarray] = None
        self.mask_goal_transformed: Optional[np.ndarray] = None
        self.mask_residual: Optional[np.ndarray] = None
        self.transform_params: Optional[TransformParams] = None
        self.optimization_cost: Optional[float] = None

    def process(self,
                image_path: str,
                goal_image_path: str,
                render: Optional[Literal['goal', 'residual', 'goal+residual']] = None,
                segmentation_prompt: str = "red Lego blocks"
                ) -> np.ndarray:
        """
        Run the full preprocessing pipeline.

        Args:
            image_path: Path to main camera image
            goal_image_path: Path to goal shape image
            render: Rendering mode
            segmentation_prompt: Text prompt for SAM3 segmentation

        Returns:
            Rendered output image
        """
        # Load input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.process_image(
            image=image,
            goal_image_path=goal_image_path,
            render=render,
            segmentation_prompt=segmentation_prompt
        )

    def process_image(self,
                      image: np.ndarray,
                      goal_image_path: str,
                      render: Optional[Literal['goal', 'residual', 'goal+residual']] = None,
                      segmentation_prompt: str = "red Lego blocks"
                      ) -> np.ndarray:
        """
        Run the full preprocessing pipeline on an image array.

        Args:
            image: Input image (H, W, 3), RGB
            goal_image_path: Path to goal shape image
            render: Rendering mode
            segmentation_prompt: Text prompt for SAM3 segmentation

        Returns:
            Rendered output image
        """
        # Step 1: Segment Lego blocks
        print("Step 1: Segmenting Lego blocks...")
        self.mask_lego = self.segmenter.segment(image, prompt=segmentation_prompt)

        # Step 2: Smooth the mask
        print("Step 2: Smoothing mask...")
        self.mask_lego_smoothed = self.smoother.smooth(self.mask_lego)

        # Step 3: Load and binarize goal image
        print("Step 3: Loading goal image...")
        self.mask_goal_binary = GoalImageProcessor.load_and_binarize(goal_image_path)

        # Step 4: Optimize transformation
        print("Step 4: Optimizing transformation...")
        self.transform_params, self.optimization_cost = self.optimizer.optimize(
            self.mask_lego_smoothed,
            self.mask_goal_binary
        )

        print(f"  Optimal transform: tx={self.transform_params.tx:.1f}, "
              f"ty={self.transform_params.ty:.1f}, "
              f"θ={np.degrees(self.transform_params.theta):.1f}°, "
              f"s={self.transform_params.scale:.2f}")
        print(f"  Final cost: {self.optimization_cost:.2f}")

        # Step 5: Transform goal mask
        self.mask_goal_transformed = GoalImageProcessor.transform_mask(
            self.mask_goal_binary,
            self.transform_params,
            self.mask_lego_smoothed.shape
        )

        # Step 6: Compute residual (pixels to remove)
        # residual = pile - goal (only where pile exists)
        self.mask_residual = (
            self.mask_lego_smoothed.astype(bool) &
            (~self.mask_goal_transformed.astype(bool))
        ).astype(np.uint8)

        # Step 7: Render output
        print("Step 5: Rendering output...")
        output = self.renderer.render(
            image,
            mask_goal=self.mask_goal_transformed,
            mask_residual=self.mask_residual,
            mode=render
        )

        return output

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown of the optimization."""
        if self.mask_goal_transformed is None:
            raise RuntimeError("Must run process() first")

        cost_fn = CostFunction(self.opt_config)
        cost_fn.set_pile_mask(self.mask_lego_smoothed)
        return cost_fn.compute_cost_breakdown(self.mask_goal_transformed)

    def test_sam_segmentation(self,
                              image_path: str,
                              segmentation_prompt: str = "red Lego blocks"
                              ) -> np.ndarray:
        """
        Test SAM segmentation and smoothing by visualizing the masks.

        Creates a 2x2 debug visualization:
        - Top-left: Original image
        - Top-right: Raw SAM segmentation mask overlaid on image
        - Bottom-left: Smoothed mask overlaid on image
        - Bottom-right: Comparison (raw mask boundary in red, smoothed in green)

        Args:
            image_path: Path to main camera image
            segmentation_prompt: Text prompt for SAM3 segmentation

        Returns:
            Debug visualization image (2x2 grid)
        """
        # Load input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.test_sam_segmentation_image(
            image=image,
            segmentation_prompt=segmentation_prompt
        )

    def test_sam_segmentation_image(self,
                                    image: np.ndarray,
                                    segmentation_prompt: str = "red Lego blocks"
                                    ) -> np.ndarray:
        """
        Test SAM segmentation and smoothing on an image array.

        Creates a 2x2 debug visualization:
        - Top-left: Original image
        - Top-right: Raw SAM segmentation mask overlaid on image
        - Bottom-left: Smoothed mask overlaid on image
        - Bottom-right: Comparison (raw mask boundary in red, smoothed in green)

        Args:
            image: Input image (H, W, 3), RGB
            segmentation_prompt: Text prompt for SAM3 segmentation

        Returns:
            Debug visualization image (2x2 grid)
        """
        h, w = image.shape[:2]

        # Step 1: Segment Lego blocks
        print("SAM Test: Segmenting Lego blocks...")
        self.mask_lego = self.segmenter.segment(image, prompt=segmentation_prompt)

        # Step 2: Smooth the mask
        print("SAM Test: Smoothing mask...")
        self.mask_lego_smoothed = self.smoother.smooth(self.mask_lego)

        # Calculate mask statistics
        raw_area = np.sum(self.mask_lego > 0)
        smoothed_area = np.sum(self.mask_lego_smoothed > 0)
        diff_area = np.sum(self.mask_lego != self.mask_lego_smoothed)

        print(f"  Raw mask area: {raw_area} pixels")
        print(f"  Smoothed mask area: {smoothed_area} pixels")
        print(f"  Difference: {diff_area} pixels ({100*diff_area/max(raw_area,1):.1f}%)")

        # Create 2x2 visualization grid
        # Calculate grid size
        grid_h, grid_w = h * 2, w * 2
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Colors for visualization
        color_raw = (255, 100, 100)      # Light red for raw mask
        color_smoothed = (100, 255, 100)  # Light green for smoothed mask
        color_raw_edge = (255, 0, 0)      # Red for raw edge
        color_smooth_edge = (0, 255, 0)   # Green for smoothed edge
        alpha = 0.4

        # Panel 1: Original image (top-left)
        panel1 = image.copy()
        self._add_label(panel1, "Original Image", (10, 30))
        grid[:h, :w] = panel1

        # Panel 2: Raw SAM mask overlay (top-right)
        panel2 = image.copy().astype(np.float32)
        overlay_raw = np.zeros_like(panel2)
        overlay_raw[self.mask_lego > 0] = color_raw
        mask_3d = np.stack([self.mask_lego > 0] * 3, axis=-1)
        panel2 = np.where(mask_3d, panel2 * (1 - alpha) + overlay_raw * alpha, panel2)
        panel2 = panel2.astype(np.uint8)
        self._add_label(panel2, f"Raw SAM Mask (area: {raw_area}px)", (10, 30))
        grid[:h, w:] = panel2

        # Panel 3: Smoothed mask overlay (bottom-left)
        panel3 = image.copy().astype(np.float32)
        overlay_smooth = np.zeros_like(panel3)
        overlay_smooth[self.mask_lego_smoothed > 0] = color_smoothed
        mask_3d = np.stack([self.mask_lego_smoothed > 0] * 3, axis=-1)
        panel3 = np.where(mask_3d, panel3 * (1 - alpha) + overlay_smooth * alpha, panel3)
        panel3 = panel3.astype(np.uint8)
        self._add_label(panel3, f"Smoothed Mask (area: {smoothed_area}px)", (10, 30))
        grid[h:, :w] = panel3

        # Panel 4: Edge comparison (bottom-right)
        panel4 = image.copy()

        # Extract edges using morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Raw mask edges
        raw_dilated = cv2.dilate(self.mask_lego, kernel, iterations=1)
        raw_eroded = cv2.erode(self.mask_lego, kernel, iterations=1)
        raw_edges = raw_dilated - raw_eroded

        # Smoothed mask edges
        smooth_dilated = cv2.dilate(self.mask_lego_smoothed, kernel, iterations=1)
        smooth_eroded = cv2.erode(self.mask_lego_smoothed, kernel, iterations=1)
        smooth_edges = smooth_dilated - smooth_eroded

        # Draw edges on panel
        panel4[raw_edges > 0] = color_raw_edge
        panel4[smooth_edges > 0] = color_smooth_edge

        self._add_label(panel4, "Edge Comparison (Red=Raw, Green=Smoothed)", (10, 30))
        grid[h:, w:] = panel4

        # Draw grid lines
        grid[h-1:h+1, :] = (128, 128, 128)
        grid[:, w-1:w+1] = (128, 128, 128)

        return grid

    def _add_label(self, image: np.ndarray, text: str, position: Tuple[int, int],
                   font_scale: float = 0.7, thickness: int = 2):
        """Add text label with background to image."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        x, y = position
        cv2.rectangle(image, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5),
                     (0, 0, 0), -1)

        # Draw text
        cv2.putText(image, text, position, font, font_scale, (255, 255, 255), thickness)


def main():
    """Example usage of the HLP Preprocessor."""
    import argparse

    parser = argparse.ArgumentParser(description='HLP Preprocessor')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--goal', type=str, required=True,
                        help='Path to goal image')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Path to output image')
    parser.add_argument('--render', type=str, default='goal+residual',
                        choices=[None, 'goal', 'residual', 'goal+residual'],
                        help='Rendering mode')
    parser.add_argument('--optimizer', type=str, default='hybrid',
                        choices=['grid', 'scipy', 'differential_evolution', 'hybrid'],
                        help='Optimizer type')
    parser.add_argument('--prompt', type=str, default='red Lego blocks',
                        help='Segmentation prompt')

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = HLPPreprocessor(optimizer_type=args.optimizer)

    # Process
    output = preprocessor.process(
        image_path=args.image,
        goal_image_path=args.goal,
        render=args.render,
        segmentation_prompt=args.prompt
    )

    # Save output
    cv2.imwrite(args.output, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Output saved to: {args.output}")

    # Print cost breakdown
    costs = preprocessor.get_cost_breakdown()
    print("\nCost breakdown:")
    for name, value in costs.items():
        print(f"  {name}: {value:.2f}")


if __name__ == '__main__':
    main()
