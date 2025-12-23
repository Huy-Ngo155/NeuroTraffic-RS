# NeuroTraffic-RS

NeuroTraffic-RS is a high-performance computer vision system built in Rust that utilizes neuromorphic principles to detect and classify over 200 types of Vietnamese traffic signs. The system architecture is inspired by the biological hierarchy of the human visual cortex, processing visual data through simulated neural layers.

## Core Architecture
The system processes information through several bio-inspired stages:
* Retina Processor: Simulates lateral inhibition and color opponency to enhance contrast and color perception.
* Visual Cortex Hierarchy:
    * V1 Layer: Specialized in edge detection using Gabor filters.
    * V2 Layer: Performs template matching for geometric shapes like Circles, Triangles, and Octagons.
    * V4 Layer: Handles color constancy and opponent color processing.
* IT Object Recognition: Extracts high-level features and manages invariant object representations.
* Spiking Neural Network (SNN): Implements Leaky Integrate-and-Fire (LIF) neurons for robust signal classification.

## Technical Features
* Comprehensive Coverage: Supports 200+ Vietnamese traffic sign classes across Warning, Prohibition, and Mandatory categories.
* Performance: Leverages the Rayon crate for multi-threaded, data-parallel computations.
* Neuromorphic Encoding: Custom vector encoding for sign shapes, colors, and hierarchical categories.
* Saliency Attention: Includes a spatial and temporal attention engine to focus on relevant visual cues.

## Tech Stack
* Language: Rust
* Computation: ndarray, Rayon
* Imaging: image crate
* Data Handling: serde, serde_json

## Getting Started
```bash
# Clone the repository
git clone [https://github.com/YourUsername/NeuroTraffic-RS.git](https://github.com/YourUsername/NeuroTraffic-RS.git)

# Build for release
cargo build --release

# Execute the vision system
cargo run
