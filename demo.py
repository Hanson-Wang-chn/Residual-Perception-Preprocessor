"""
Demo script for HLP Preprocessor

This script demonstrates how to use the HLP Preprocessor module for the
"Sweep to Shapes" manipulation task.

Usage:
    python demo.py --image <path_to_image> --goal <path_to_goal> [options]
    python demo.py --config config.yaml [options]

Example:
    python demo.py --image input.jpg --goal data/E.png --render goal+residual --output result.png
    python demo.py --config config/config.yaml --render goal+residual
"""

import argparse
import cv2
import numpy as np
import os
import sys
import yaml
from typing import Dict, Any, Optional

from hlp_preprocessor import (
    HLPPreprocessor,
    OptimizationConfig,
    SmoothingConfig,
    create_optimizer,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='HLP Preprocessor Demo for Sweep to Shapes task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python demo.py --image lego_pile.jpg --goal data/E.png --output result.png

  # Use config file
  python demo.py --config config/config.yaml

  # Use config file with command line override
  python demo.py --config config/config.yaml --render goal --optimizer grid

  # Use grid search optimizer for more precise (but slower) optimization
  python demo.py --image lego_pile.jpg --goal data/Z.png --optimizer grid

  # Customize rendering colors
  python demo.py --image lego_pile.jpg --goal data/N.png --color-goal 255,255,0 --color-residual 0,255,255

  # Custom optimization weights
  python demo.py --image lego_pile.jpg --goal data/E.png --lambda-fill 1.5 --lambda-sweep 2.0

  # Different render modes
  python demo.py --image lego_pile.jpg --goal data/E.png --render goal
  python demo.py --image lego_pile.jpg --goal data/E.png --render residual
  python demo.py --image lego_pile.jpg --goal data/E.png --render goal+residual
        """
    )

    # Config file
    parser.add_argument('--config', type=str, default="config/config.yaml",
                        help='Path to YAML config file (optional)')

    # Required arguments (can be optional if config is provided)
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (main camera view)')
    parser.add_argument('--goal', type=str, default=None,
                        help='Path to goal shape image (binary or grayscale)')

    # Output settings
    parser.add_argument('--output', type=str, default='output.png',
                        help='Path to save output image (default: output.png)')
    parser.add_argument('--render', type=str, default='goal+residual',
                        choices=['none', 'goal', 'residual', 'goal+residual'],
                        help='Rendering mode (default: goal+residual)')

    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='hybrid',
                        choices=['grid', 'scipy', 'differential_evolution', 'hybrid'],
                        help='Optimization algorithm (default: hybrid)')
    parser.add_argument('--grid-resolution', type=int, default=20,
                        help='Grid search resolution (default: 20)')

    # Cost function weights
    parser.add_argument('--lambda-fill', type=float, default=1.0,
                        help='Weight for fill cost (default: 1.0)')
    parser.add_argument('--lambda-remove', type=float, default=1.0,
                        help='Weight for remove cost (default: 1.0)')
    parser.add_argument('--lambda-edge', type=float, default=0.5,
                        help='Weight for edge alignment (default: 0.5)')
    parser.add_argument('--lambda-sweep', type=float, default=1.0,
                        help='Weight for sweep difficulty (default: 1.0)')

    # Other optimization parameters
    parser.add_argument('--sigma-edge', type=float, default=10.0,
                        help='Sigma for edge cost Gaussian (default: 10.0)')
    parser.add_argument('--alpha-sweep', type=float, default=2.0,
                        help='Exponent for sweep cost distance field (default: 2.0)')
    parser.add_argument('--scale-min', type=float, default=0.3,
                        help='Minimum scale factor (default: 0.3)')
    parser.add_argument('--scale-max', type=float, default=1.5,
                        help='Maximum scale factor (default: 1.5)')

    # Rendering colors (RGB)
    parser.add_argument('--color-goal', type=str, default='0,255,0',
                        help='RGB color for goal overlay (default: 0,255,0 = green)')
    parser.add_argument('--color-residual', type=str, default='255,0,255',
                        help='RGB color for residual overlay (default: 255,0,255 = magenta)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Overlay transparency 0-1 (default: 0.5)')

    # Smoothing parameters
    parser.add_argument('--smooth-kernel', type=int, default=5,
                        help='Morphological kernel size (default: 5)')
    parser.add_argument('--smooth-sigma', type=float, default=2.0,
                        help='Gaussian smoothing sigma (default: 2.0)')

    # SAM3 parameters
    parser.add_argument('--prompt', type=str, default='red Lego blocks',
                        help='Text prompt for SAM3 segmentation (default: "red Lego blocks")')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='SAM3 confidence threshold (default: 0.3)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for SAM3 (default: cuda)')

    # Debug options
    parser.add_argument('--test-sam', action='store_true',
                        help='Test SAM segmentation mode: output mask overlay and skip optimization')

    # Additional options
    parser.add_argument('--save-masks', action='store_true',
                        help='Save intermediate masks')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    return parser


def merge_configs(yaml_config: Dict[str, Any], cmd_args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge YAML config with command line arguments.
    Command line arguments override YAML config when explicitly provided.
    """
    # Start with command line arguments
    merged = argparse.Namespace(**vars(cmd_args))
    
    # Get the parser to know defaults
    parser = create_parser()
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help':
            defaults[action.dest] = action.default
    
    # For each YAML config value, use it if command line arg is still at default
    for key, value in yaml_config.items():
        # Convert hyphenated keys to underscored (YAML uses hyphens, argparse uses underscores)
        key_underscore = key.replace('-', '_')
        
        cmd_value = getattr(merged, key_underscore, None)
        default_value = defaults.get(key_underscore)
        
        # If command line value equals default, use YAML value
        if cmd_value == default_value or cmd_value is None:
            setattr(merged, key_underscore, value)
    
    return merged


def parse_args():
    """Parse command line arguments and merge with config file if provided."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If config file is provided, load and merge
    if args.config is not None:
        yaml_config = load_config(args.config)
        if yaml_config:
            args = merge_configs(yaml_config, args)
    
    return args


def parse_color(color_str):
    """Parse color string like '255,0,128' to tuple."""
    if isinstance(color_str, (list, tuple)):
        if len(color_str) != 3:
            raise ValueError(f"Invalid color format: {color_str}. Must have 3 values (R,G,B).")
        return tuple(int(c) for c in color_str)
    
    parts = str(color_str).split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid color format: {color_str}. Use R,G,B format.")
    return tuple(int(p.strip()) for p in parts)


def validate_args(args):
    """Validate required arguments."""
    if args.image is None:
        print("Error: --image is required (or specify in config file)")
        sys.exit(1)

    # Goal is only required when not in test_sam mode
    if not args.test_sam and args.goal is None:
        print("Error: --goal is required (or specify in config file)")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        sys.exit(1)
    if args.goal is not None and not os.path.exists(args.goal):
        print(f"Error: Goal image not found: {args.goal}")
        sys.exit(1)


def main():
    args = parse_args()
    
    # Validate inputs
    validate_args(args)

    # Parse colors
    try:
        color_goal = parse_color(args.color_goal)
        color_residual = parse_color(args.color_residual)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create configuration
    opt_config = OptimizationConfig(
        lambda_fill=args.lambda_fill,
        lambda_remove=args.lambda_remove,
        lambda_edge=args.lambda_edge,
        lambda_sweep=args.lambda_sweep,
        sigma_edge=args.sigma_edge,
        alpha_sweep=args.alpha_sweep,
        grid_resolution=args.grid_resolution,
        scale_range=(args.scale_min, args.scale_max),
    )

    smooth_config = SmoothingConfig(
        kernel_size=args.smooth_kernel,
        gaussian_sigma=args.smooth_sigma,
    )

    render_config = {
        'color_goal': color_goal,
        'color_residual': color_residual,
        'alpha': args.alpha,
    }

    if args.verbose:
        print("=" * 60)
        print("Configuration:")
        print("=" * 60)
        if args.config:
            print(f"Config file: {args.config}")
        print(f"Input image: {args.image}")
        if not args.test_sam:
            print(f"Goal image: {args.goal}")
        print(f"Output: {args.output}")
        if args.test_sam:
            print(f"\n*** TEST SAM MODE ***")
            print(f"Smoothing kernel: {args.smooth_kernel}")
            print(f"Smoothing sigma: {args.smooth_sigma}")
        else:
            print(f"\nOptimizer: {args.optimizer}")
            print(f"Lambda weights:")
            print(f"  fill={args.lambda_fill}, remove={args.lambda_remove}")
            print(f"  edge={args.lambda_edge}, sweep={args.lambda_sweep}")
            print(f"Scale range: [{args.scale_min}, {args.scale_max}]")
            print(f"Render mode: {args.render}")
            print(f"Colors: goal={color_goal}, residual={color_residual}")
        print(f"Segmentation prompt: '{args.prompt}'")
        print(f"Device: {args.device}")
        print("=" * 60)
        print()

    # Create preprocessor
    from hlp_preprocessor import LegoSegmenter, MaskSmoother

    segmenter = LegoSegmenter(
        device=args.device,
        confidence_threshold=args.confidence
    )
    smoother = MaskSmoother(smooth_config)

    preprocessor = HLPPreprocessor(
        segmenter=segmenter,
        smoother=smoother,
        optimization_config=opt_config,
        optimizer_type=args.optimizer,
        render_config=render_config,
    )

    # Test SAM mode: only run segmentation and smoothing, output debug visualization
    if args.test_sam:
        print("Running SAM segmentation test...")
        output = preprocessor.test_sam_segmentation(
            image_path=args.image,
            segmentation_prompt=args.prompt,
        )

        # Save output
        cv2.imwrite(args.output, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print(f"\n✓ SAM test output saved to: {args.output}")
        print("  Visualization shows 2x2 grid:")
        print("    Top-left:     Original image")
        print("    Top-right:    Raw SAM mask (red overlay)")
        print("    Bottom-left:  Smoothed mask (green overlay)")
        print("    Bottom-right: Edge comparison (red=raw, green=smoothed)")

        # Save intermediate masks if requested
        if args.save_masks:
            base_name = os.path.splitext(args.output)[0]

            # Save raw Lego mask
            raw_mask_path = f"{base_name}_mask_raw.png"
            cv2.imwrite(raw_mask_path, preprocessor.mask_lego * 255)
            print(f"\n✓ Raw SAM mask saved to: {raw_mask_path}")

            # Save smoothed Lego mask
            smooth_mask_path = f"{base_name}_mask_smoothed.png"
            cv2.imwrite(smooth_mask_path, preprocessor.mask_lego_smoothed * 255)
            print(f"✓ Smoothed mask saved to: {smooth_mask_path}")

        return

    # Normal processing mode
    print("Processing image...")
    render_mode = None if args.render == 'none' else args.render
    output = preprocessor.process(
        image_path=args.image,
        goal_image_path=args.goal,
        render=render_mode,
        segmentation_prompt=args.prompt,
    )

    # Save output
    cv2.imwrite(args.output, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"\n✓ Output saved to: {args.output}")

    # Print results
    print(f"\nOptimal transformation:")
    print(f"  Translation: ({preprocessor.transform_params.tx:.1f}, {preprocessor.transform_params.ty:.1f})")
    print(f"  Rotation: {np.degrees(preprocessor.transform_params.theta):.1f}°")
    print(f"  Scale: {preprocessor.transform_params.scale:.2f}")

    # Print cost breakdown
    costs = preprocessor.get_cost_breakdown()
    print(f"\nCost breakdown:")
    print(f"  Fill cost:   {costs['fill']:.0f} (weighted: {args.lambda_fill * costs['fill']:.0f})")
    print(f"  Remove cost: {costs['remove']:.0f} (weighted: {args.lambda_remove * costs['remove']:.0f})")
    print(f"  Edge cost:   {costs['edge']:.1f} (weighted: {args.lambda_edge * costs['edge']:.1f})")
    print(f"  Sweep cost:  {costs['sweep']:.0f} (weighted: {args.lambda_sweep * costs['sweep']:.0f})")
    print(f"  Total cost:  {costs['total']:.0f}")

    # Save intermediate masks if requested
    if args.save_masks:
        base_name = os.path.splitext(args.output)[0]

        # Save Lego mask
        lego_mask_path = f"{base_name}_mask_lego.png"
        cv2.imwrite(lego_mask_path, preprocessor.mask_lego_smoothed * 255)
        print(f"\n✓ Lego mask saved to: {lego_mask_path}")

        # Save goal mask
        goal_mask_path = f"{base_name}_mask_goal.png"
        cv2.imwrite(goal_mask_path, preprocessor.mask_goal_transformed * 255)
        print(f"✓ Goal mask saved to: {goal_mask_path}")

        # Save residual mask
        residual_mask_path = f"{base_name}_mask_residual.png"
        cv2.imwrite(residual_mask_path, preprocessor.mask_residual * 255)
        print(f"✓ Residual mask saved to: {residual_mask_path}")


if __name__ == '__main__':
    main()
