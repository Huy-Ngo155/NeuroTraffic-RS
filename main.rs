use std::collections::{HashMap, VecDeque, HashSet};
use std::f32::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use image::{self, DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, Axis, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use lazy_static::lazy_static;
use regex::Regex;
use ordered_float::OrderedFloat;
use indicatif::{ProgressBar, ProgressStyle};

const TOTAL_SIGN_CLASSES: usize = 200;
const FEATURE_VECTOR_SIZE: usize = 512;
const MAX_TRAINING_IMAGES: usize = 10000;
const MINI_BATCH_SIZE: usize = 32;
const EPOCHS: usize = 50;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VietnameseTrafficSign {
    pub id: u32,
    pub vietnamese_name: String,
    pub english_name: String,
    pub category: SignCategory,
    pub shape: SignShape,
    pub color: SignColor,
    pub description: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SignCategory {
    Warning,
    Prohibition,
    Mandatory,
    Information,
    Priority,
    Temporary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SignShape {
    Circle,
    Triangle,
    Rectangle,
    Square,
    Octagon,
    Diamond,
    Pentagon,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SignColor {
    Red,
    Blue,
    Yellow,
    White,
    Black,
    Green,
    Orange,
    Brown,
    MultiColor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignDataset {
    pub signs: HashMap<u32, VietnameseTrafficSign>,
    pub images: HashMap<u32, Vec<ImageData>>,
    pub statistics: DatasetStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub sign_id: u32,
    pub variations: Vec<Variation>,
    pub annotations: Vec<Annotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variation {
    pub rotation: f32,
    pub scale: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub blur: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub verified: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_images: usize,
    pub per_category: HashMap<SignCategory, usize>,
    pub per_shape: HashMap<SignShape, usize>,
    pub per_color: HashMap<SignColor, usize>,
    pub avg_image_size: (f32, f32),
}

pub struct NeuromorphicVisionSystem {
    retina_processor: RetinaProcessor,
    visual_cortex: VisualCortex,
    attention_engine: AttentionEngine,
    memory_network: MemoryNetwork,
    classifier: NeuroClassifier,
    sign_database: Arc<RwLock<SignDatabase>>,
    config: SystemConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetinaProcessor {
    layers: Vec<RetinaLayer>,
    lateral_inhibition: bool,
    color_opponency: bool,
    motion_sensitive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetinaLayer {
    pub cells: Array3<f32>,
    pub weights: Array3<f32>,
    pub biases: Array2<f32>,
    pub activation: ActivationFunction,
    pub receptive_field: ReceptiveField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualCortex {
    pub v1_layers: Vec<V1Layer>,
    pub v2_complex: V2ComplexCells,
    pub v4_color: V4ColorProcessing,
    pub it_object: ITObjectRecognition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1Layer {
    pub orientation_maps: Array3<f32>,
    pub spatial_frequency: Array2<f32>,
    pub simple_cells: Vec<GaborFilter>,
    pub complex_cells: Vec<ComplexCell>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaborFilter {
    pub kernel: Array2<f32>,
    pub theta: f32,
    pub sigma: f32,
    pub lambda: f32,
    pub gamma: f32,
    pub phase: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexCell {
    pub pooling_size: (usize, usize),
    pub max_pooling: bool,
    pub normalization: NormalizationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2ComplexCells {
    pub shape_templates: HashMap<SignShape, Array2<f32>>,
    pub contour_integrators: Array3<f32>,
    pub symmetry_detectors: Array3<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V4ColorProcessing {
    pub color_channels: Array3<f32>,
    pub color_constancy: bool,
    pub opponent_colors: OpponentColorSpace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ITObjectRecognition {
    pub feature_hierarchy: FeatureHierarchy,
    pub invariant_features: Array1<f32>,
    pub object_prototypes: HashMap<u32, ObjectPrototype>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionEngine {
    pub saliency_maps: Array2<f32>,
    pub spatial_attention: SpatialAttention,
    pub feature_attention: FeatureAttention,
    pub temporal_attention: TemporalAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNetwork {
    pub working_memory: WorkingMemory,
    pub episodic_memory: EpisodicMemory,
    pub semantic_memory: SemanticMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroClassifier {
    pub spiking_network: SpikingNeuralNetwork,
    pub decision_layers: Vec<DecisionLayer>,
    pub confidence_calibration: ConfidenceCalibration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignDatabase {
    pub signs: HashMap<u32, VietnameseTrafficSign>,
    pub features: HashMap<u32, Array1<f32>>,
    pub embeddings: HashMap<u32, Array1<f32>>,
    pub relationships: HashMap<u32, Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub min_detection_confidence: f32,
    pub min_classification_confidence: f32,
    pub max_detections_per_image: usize,
    pub image_size: (u32, u32),
    pub enable_augmentation: bool,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub dropout_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softplus,
    Spiking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptiveField {
    pub center: (f32, f32),
    pub surround: (f32, f32),
    pub sigma_center: f32,
    pub sigma_surround: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpponentColorSpace {
    pub red_green: Array2<f32>,
    pub blue_yellow: Array2<f32>,
    pub black_white: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureHierarchy {
    pub low_level: Vec<String>,
    pub mid_level: Vec<String>,
    pub high_level: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectPrototype {
    pub id: u32,
    pub features: Array1<f32>,
    pub variance: Array1<f32>,
    pub frequency: u32,
    pub last_seen: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAttention {
    pub focus_map: Array2<f32>,
    pub inhibition_return: bool,
    pub spotlight_size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAttention {
    pub feature_weights: Array1<f32>,
    pub channel_attention: Array1<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAttention {
    pub temporal_filter: Array1<f32>,
    pub decay_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub capacity: usize,
    pub items: VecDeque<MemoryItem>,
    pub decay: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub episodes: Vec<Episode>,
    pub consolidation_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    pub concepts: HashMap<String, Concept>,
    pub associations: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingNeuralNetwork {
    pub neurons: Vec<LIFNeuron>,
    pub synapses: Vec<Synapse>,
    pub layers: Vec<SNNLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: ActivationFunction,
    pub dropout: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceCalibration {
    pub temperature: f32,
    pub bins: usize,
    pub histogram: Array1<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: u32,
    pub features: Array1<f32>,
    pub timestamp: i64,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub sequence: Vec<u32>,
    pub context: HashMap<String, f32>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub features: Array1<f32>,
    pub instances: Vec<u32>,
    pub prototypicality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFNeuron {
    pub membrane_potential: f32,
    pub threshold: f32,
    pub reset_potential: f32,
    pub tau_m: f32,
    pub tau_s: f32,
    pub refractory_period: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub weight: f32,
    pub delay: u32,
    pub plasticity: SynapticPlasticity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SNNLayer {
    pub neurons: Vec<LIFNeuron>,
    pub connectivity: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticPlasticity {
    pub stdp_rate: f32,
    pub ltp: f32,
    pub ltd: f32,
    pub homeostatic_scaling: bool,
}

impl NeuromorphicVisionSystem {
    pub fn new(config: SystemConfig) -> Self {
        let retina_processor = RetinaProcessor::new();
        let visual_cortex = VisualCortex::new();
        let attention_engine = AttentionEngine::new();
        let memory_network = MemoryNetwork::new();
        let classifier = NeuroClassifier::new(TOTAL_SIGN_CLASSES);
        let sign_database = Arc::new(RwLock::new(SignDatabase::new()));
        
        Self {
            retina_processor,
            visual_cortex,
            attention_engine,
            memory_network,
            classifier,
            sign_database,
            config,
        }
    }
    
    pub fn load_vietnamese_signs(&mut self, database_path: &Path) -> Result<usize, String> {
        let pb = ProgressBar::new(200);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap());
        
        let signs = VietnameseTrafficSign::load_all_signs(database_path)?;
        let mut db = self.sign_database.write().unwrap();
        
        for (id, sign) in signs {
            db.signs.insert(id, sign.clone());
            
            let features = self.extract_prototypical_features(&sign);
            db.features.insert(id, features);
            
            pb.inc(1);
        }
        
        pb.finish_with_message("Loaded all Vietnamese traffic signs");
        Ok(db.signs.len())
    }
    
    pub fn train(&mut self, dataset: &SignDataset) -> Result<TrainingStats, String> {
        println!("Starting training for {} traffic sign classes...", TOTAL_SIGN_CLASSES);
        
        let mut stats = TrainingStats::new();
        let mut all_features = Vec::new();
        let mut all_labels = Vec::new();
        
        let pb = ProgressBar::new(dataset.images.len() as u64);
        
        for (sign_id, images) in &dataset.images {
            for image_data in images {
                let img = image::open(&image_data.path)
                    .map_err(|e| format!("Failed to open image: {}", e))?;
                
                let processed = self.process_image(&img);
                let features = self.extract_features(&processed);
                
                all_features.push(features);
                all_labels.push(*sign_id);
                
                pb.inc(1);
            }
        }
        
        pb.finish_with_message("Feature extraction completed");
        
        println!("Training neuro-classifier with {} samples...", all_features.len());
        
        let training_result = self.classifier.train(
            &all_features,
            &all_labels,
            self.config.learning_rate,
            self.config.momentum,
        );
        
        stats.accuracy = training_result.accuracy;
        stats.loss = training_result.loss;
        stats.training_time = training_result.training_time;
        
        println!("Training completed. Accuracy: {:.2}%", stats.accuracy * 100.0);
        
        Ok(stats)
    }
    
    pub fn detect_and_classify(&self, image: &DynamicImage) -> Vec<DetectionResult> {
        let start_time = Instant::now();
        
        let retina_output = self.retina_processor.process(image);
        let cortex_output = self.visual_cortex.process(&retina_output);
        let attention_map = self.attention_engine.compute(&cortex_output);
        let proposals = self.generate_proposals(&cortex_output, &attention_map);
        
        let mut detections = Vec::new();
        
        for proposal in proposals {
            if proposal.confidence < self.config.min_detection_confidence {
                continue;
            }
            
            let cropped = self.crop_proposal(image, &proposal.bbox);
            let features = self.extract_features_from_crop(&cropped);
            let classification = self.classifier.classify(&features);
            
            if classification.confidence >= self.config.min_classification_confidence {
                let mut db = self.sign_database.write().unwrap();
                if let Some(sign) = db.signs.get(&classification.sign_id) {
                    detections.push(DetectionResult {
                        sign_id: classification.sign_id,
                        sign: sign.clone(),
                        bbox: proposal.bbox,
                        detection_confidence: proposal.confidence,
                        classification_confidence: classification.confidence,
                        features: features.clone(),
                        processing_time: start_time.elapsed().as_millis() as f32 / 1000.0,
                    });
                }
            }
            
            if detections.len() >= self.config.max_detections_per_image {
                break;
            }
        }
        
        detections.sort_by(|a, b| {
            b.detection_confidence
                .partial_cmp(&a.detection_confidence)
                .unwrap()
        });
        
        detections
    }
    
    pub fn process_image_batch(
        &self,
        images: Vec<DynamicImage>,
    ) -> Vec<Vec<DetectionResult>> {
        images
            .par_iter()
            .map(|img| self.detect_and_classify(img))
            .collect()
    }
    
    pub fn save_model(&self, path: &Path) -> Result<(), String> {
        let model_data = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;
        
        fs::write(path, model_data)
            .map_err(|e| format!("Failed to write model file: {}", e))?;
        
        Ok(())
    }
    
    pub fn load_model(path: &Path) -> Result<Self, String> {
        let data = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read model file: {}", e))?;
        
        let system: Self = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;
        
        Ok(system)
    }
    
    fn extract_prototypical_features(&self, sign: &VietnameseTrafficSign) -> Array1<f32> {
        let mut features = Array1::zeros(FEATURE_VECTOR_SIZE);
        
        let shape_code = self.encode_shape(&sign.shape);
        let color_code = self.encode_color(&sign.color);
        let category_code = self.encode_category(&sign.category);
        
        for i in 0..shape_code.len() {
            features[i] = shape_code[i];
        }
        
        let offset = shape_code.len();
        for i in 0..color_code.len() {
            features[offset + i] = color_code[i];
        }
        
        let offset2 = offset + color_code.len();
        for i in 0..category_code.len() {
            features[offset2 + i] = category_code[i];
        }
        
        let name_embedding = self.embed_text(&sign.vietnamese_name);
        let desc_embedding = self.embed_text(&sign.description);
        
        let offset3 = offset2 + category_code.len();
        for i in 0..name_embedding.len() {
            features[offset3 + i] = name_embedding[i];
        }
        
        let offset4 = offset3 + name_embedding.len();
        for i in 0..desc_embedding.len().min(features.len() - offset4) {
            features[offset4 + i] = desc_embedding[i];
        }
        
        features
    }
    
    fn encode_shape(&self, shape: &SignShape) -> Vec<f32> {
        match shape {
            SignShape::Circle => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignShape::Triangle => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignShape::Rectangle => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            SignShape::Square => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            SignShape::Octagon => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            SignShape::Diamond => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            SignShape::Pentagon => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
    
    fn encode_color(&self, color: &SignColor) -> Vec<f32> {
        match color {
            SignColor::Red => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignColor::Blue => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignColor::Yellow => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignColor::White => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignColor::Black => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            SignColor::Green => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            SignColor::Orange => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            SignColor::Brown => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            SignColor::MultiColor => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
    
    fn encode_category(&self, category: &SignCategory) -> Vec<f32> {
        match category {
            SignCategory::Warning => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SignCategory::Prohibition => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            SignCategory::Mandatory => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            SignCategory::Information => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            SignCategory::Priority => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            SignCategory::Temporary => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
    
    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; 64];
        let chars: Vec<char> = text.chars().collect();
        
        for (i, ch) in chars.iter().enumerate().take(64) {
            embedding[i] = (*ch as u32) as f32 / 65535.0;
        }
        
        embedding
    }
}

impl VietnameseTrafficSign {
    pub fn load_all_signs(database_path: &Path) -> Result<HashMap<u32, Self>, String> {
        let mut signs = HashMap::new();
        
        let categories = [
            ("warning", SignCategory::Warning),
            ("prohibition", SignCategory::Prohibition),
            ("mandatory", SignCategory::Mandatory),
            ("information", SignCategory::Information),
            ("priority", SignCategory::Priority),
            ("temporary", SignCategory::Temporary),
        ];
        
        let mut current_id = 1;
        
        for (category_name, category) in categories.iter() {
            let category_path = database_path.join(category_name);
            
            if category_path.exists() {
                if let Ok(entries) = fs::read_dir(category_path) {
                    for entry in entries.flatten() {
                        if entry.path().is_file() {
                            let sign = Self::from_file(&entry.path(), *category, current_id)?;
                            signs.insert(current_id, sign);
                            current_id += 1;
                            
                            if current_id > 200 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(signs)
    }
    
    fn from_file(path: &Path, category: SignCategory, id: u32) -> Result<Self, String> {
        let file_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        let parts: Vec<&str> = file_name.split('_').collect();
        let vietnamese_name = parts.get(0).unwrap_or(&"unknown").to_string();
        let english_name = parts.get(1).unwrap_or(&"unknown").to_string();
        
        let shape = Self::determine_shape_from_name(&vietnamese_name);
        let color = Self::determine_color_from_name(&vietnamese_name);
        let priority = Self::determine_priority(&category, &vietnamese_name);
        
        Ok(Self {
            id,
            vietnamese_name,
            english_name,
            category,
            shape,
            color,
            description: Self::generate_description(id, &category),
            priority,
        })
    }
    
    fn determine_shape_from_name(name: &str) -> SignShape {
        if name.contains("tron") || name.contains("hình tròn") {
            SignShape::Circle
        } else if name.contains("tam giác") || name.contains("hình tam giác") {
            SignShape::Triangle
        } else if name.contains("vuông") {
            SignShape::Square
        } else if name.contains("chữ nhật") {
            SignShape::Rectangle
        } else if name.contains("bát giác") {
            SignShape::Octagon
        } else if name.contains("hình thoi") {
            SignShape::Diamond
        } else {
            SignShape::Circle
        }
    }
    
    fn determine_color_from_name(name: &str) -> SignColor {
        if name.contains("đỏ") {
            SignColor::Red
        } else if name.contains("xanh") {
            SignColor::Blue
        } else if name.contains("vàng") {
            SignColor::Yellow
        } else if name.contains("trắng") {
            SignColor::White
        } else if name.contains("đen") {
            SignColor::Black
        } else if name.contains("xanh lá") {
            SignColor::Green
        } else if name.contains("cam") {
            SignColor::Orange
        } else if name.contains("nâu") {
            SignColor::Brown
        } else {
            SignColor::MultiColor
        }
    }
    
    fn determine_priority(category: &SignCategory, name: &str) -> u8 {
        match category {
            SignCategory::Priority => 10,
            SignCategory::Prohibition => 9,
            SignCategory::Mandatory => 8,
            SignCategory::Warning => 7,
            SignCategory::Temporary => 6,
            SignCategory::Information => 5,
        }
    }
    
    fn generate_description(id: u32, category: &SignCategory) -> String {
        format!("Biển báo giao thông Việt Nam ID {} - Loại {:?}", id, category)
    }
}

#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub sign_id: u32,
    pub sign: VietnameseTrafficSign,
    pub bbox: BoundingBox,
    pub detection_confidence: f32,
    pub classification_confidence: f32,
    pub features: Array1<f32>,
    pub processing_time: f32,
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub accuracy: f32,
    pub loss: f32,
    pub training_time: f64,
    pub epoch_stats: Vec<EpochStat>,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            loss: 0.0,
            training_time: 0.0,
            epoch_stats: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpochStat {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_accuracy: f32,
    pub learning_rate: f32,
}

pub struct Proposal {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub features: Array1<f32>,
}

struct ClassificationResult {
    pub sign_id: u32,
    pub confidence: f32,
    pub probabilities: Array1<f32>,
}

struct TrainingResult {
    pub accuracy: f32,
    pub loss: f32,
    pub training_time: f64,
}

impl RetinaProcessor {
    pub fn new() -> Self {
        Self {
            layers: Self::create_retina_layers(),
            lateral_inhibition: true,
            color_opponency: true,
            motion_sensitive: false,
        }
    }
    
    fn create_retina_layers() -> Vec<RetinaLayer> {
        let mut layers = Vec::new();
        
        layers.push(RetinaLayer {
            cells: Array3::zeros((64, 64, 3)),
            weights: Array3::from_elem((3, 3, 3), 0.1),
            biases: Array2::zeros((64, 64)),
            activation: ActivationFunction::ReLU,
            receptive_field: ReceptiveField {
                center: (0.5, 0.5),
                surround: (1.5, 1.5),
                sigma_center: 0.5,
                sigma_surround: 1.5,
            },
        });
        
        layers
    }
    
    pub fn process(&self, image: &DynamicImage) -> Array3<f32> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        let mut output = Array3::zeros((height as usize, width as usize, 3));
        
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                output[[y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0;
                output[[y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
                output[[y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
            }
        }
        
        if self.lateral_inhibition {
            output = self.apply_lateral_inhibition(&output);
        }
        
        if self.color_opponency {
            output = self.apply_color_opponency(&output);
        }
        
        output
    }
    
    fn apply_lateral_inhibition(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = input.clone();
        
        let inhibition_kernel = Array2::from_shape_fn((5, 5), |(i, j)| {
            let x = i as f32 - 2.0;
            let y = j as f32 - 2.0;
            (-(x * x + y * y) / 8.0).exp() * -0.1
        });
        
        for ch in 0..c {
            let channel = input.slice(s![.., .., ch]);
            let inhibited = Self::convolve_2d(&channel.view(), &inhibition_kernel.view());
            
            for i in 0..h {
                for j in 0..w {
                    let val = input[[i, j, ch]] + inhibited[[i, j]];
                    output[[i, j, ch]] = val.max(0.0);
                }
            }
        }
        
        output
    }
    
    fn apply_color_opponency(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((h, w, 3));
        
        for i in 0..h {
            for j in 0..w {
                let r = input[[i, j, 0]];
                let g = input[[i, j, 1]];
                let b = input[[i, j, 2]];
                
                output[[i, j, 0]] = r - g;
                output[[i, j, 1]] = b - (r + g) / 2.0;
                output[[i, j, 2]] = (r + g + b) / 3.0;
            }
        }
        
        output
    }
    
    fn convolve_2d(input: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
        let (h, w) = (input.shape()[0], input.shape()[1]);
        let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
        let mut output = Array2::zeros((h, w));
        
        for i in kh/2..h - kh/2 {
            for j in kw/2..w - kw/2 {
                let mut sum = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += input[[i + ki - kh/2, j + kj - kw/2]] * kernel[[ki, kj]];
                    }
                }
                output[[i, j]] = sum;
            }
        }
        
        output
    }
}

impl VisualCortex {
    pub fn new() -> Self {
        Self {
            v1_layers: Self::create_v1_layers(),
            v2_complex: V2ComplexCells::new(),
            v4_color: V4ColorProcessing::new(),
            it_object: ITObjectRecognition::new(),
        }
    }
    
    fn create_v1_layers() -> Vec<V1Layer> {
        let mut layers = Vec::new();
        
        let orientations = [0.0, 45.0, 90.0, 135.0];
        let spatial_frequencies = [2.0, 4.0, 8.0];
        
        for sf in &spatial_frequencies {
            let mut simple_cells = Vec::new();
            
            for &theta in &orientations {
                let gabor = GaborFilter {
                    kernel: Array2::zeros((7, 7)),
                    theta,
                    sigma: 2.0,
                    lambda: *sf,
                    gamma: 0.5,
                    phase: 0.0,
                };
                simple_cells.push(gabor);
            }
            
            let layer = V1Layer {
                orientation_maps: Array3::zeros((64, 64, orientations.len())),
                spatial_frequency: Array2::zeros((64, 64)),
                simple_cells,
                complex_cells: vec![
                    ComplexCell {
                        pooling_size: (2, 2),
                        max_pooling: true,
                        normalization: NormalizationType::BatchNorm,
                    }
                ],
            };
            
            layers.push(layer);
        }
        
        layers
    }
    
    pub fn process(&self, input: &Array3<f32>) -> Array3<f32> {
        let v1_output = self.process_v1(input);
        let v2_output = self.v2_complex.process(&v1_output);
        let v4_output = self.v4_color.process(&v2_output);
        let it_output = self.it_object.process(&v4_output);
        
        it_output
    }
    
    fn process_v1(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((h, w, self.v1_layers.len()));
        
        for (l, layer) in self.v1_layers.iter().enumerate() {
            for (c, gabor) in layer.simple_cells.iter().enumerate() {
                let response = Self::apply_gabor(input, gabor);
                
                for i in 0..h {
                    for j in 0..w {
                        output[[i, j, l]] += response[[i, j]];
                    }
                }
            }
        }
        
        output
    }
    
    fn apply_gabor(input: &Array3<f32>, gabor: &GaborFilter) -> Array2<f32> {
        let gray = Self::rgb_to_gray(input);
        let kernel = Self::generate_gabor_kernel(gabor);
        
        let (h, w) = (gray.shape()[0], gray.shape()[1]);
        let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
        let mut output = Array2::zeros((h - kh + 1, w - kw + 1));
        
        for i in 0..h - kh + 1 {
            for j in 0..w - kw + 1 {
                let mut sum = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += gray[[i + ki, j + kj]] * kernel[[ki, kj]];
                    }
                }
                output[[i, j]] = sum;
            }
        }
        
        output
    }
    
    fn generate_gabor_kernel(gabor: &GaborFilter) -> Array2<f32> {
        let size = 7;
        let center = size as f32 / 2.0;
        let mut kernel = Array2::zeros((size, size));
        
        let theta_rad = gabor.theta * PI / 180.0;
        let cos_theta = theta_rad.cos();
        let sin_theta = theta_rad.sin();
        
        for i in 0..size {
            for j in 0..size {
                let x = i as f32 - center;
                let y = j as f32 - center;
                
                let x_theta = x * cos_theta + y * sin_theta;
                let y_theta = -x * sin_theta + y * cos_theta;
                
                let exp_term = (-(x_theta.powi(2) + gabor.gamma.powi(2) * y_theta.powi(2))
                    / (2.0 * gabor.sigma.powi(2)))
                    .exp();
                let cos_term = (2.0 * PI * x_theta / gabor.lambda + gabor.phase).cos();
                
                kernel[[i, j]] = exp_term * cos_term;
            }
        }
        
        kernel
    }
    
    fn rgb_to_gray(rgb: &Array3<f32>) -> Array2<f32> {
        let (h, w, _) = (rgb.shape()[0], rgb.shape()[1], rgb.shape()[2]);
        let mut gray = Array2::zeros((h, w));
        
        for i in 0..h {
            for j in 0..w {
                gray[[i, j]] = 0.299 * rgb[[i, j, 0]]
                    + 0.587 * rgb[[i, j, 1]]
                    + 0.114 * rgb[[i, j, 2]];
            }
        }
        
        gray
    }
}

impl V2ComplexCells {
    pub fn new() -> Self {
        let mut shape_templates = HashMap::new();
        
        for shape in &[
            SignShape::Circle,
            SignShape::Triangle,
            SignShape::Rectangle,
            SignShape::Square,
            SignShape::Octagon,
            SignShape::Diamond,
            SignShape::Pentagon,
        ] {
            let template = Self::create_shape_template(shape, 15);
            shape_templates.insert(shape.clone(), template);
        }
        
        Self {
            shape_templates,
            contour_integrators: Array3::zeros((64, 64, 8)),
            symmetry_detectors: Array3::zeros((64, 64, 4)),
        }
    }
    
    fn create_shape_template(shape: &SignShape, size: usize) -> Array2<f32> {
        let mut template = Array2::zeros((size, size));
        let center = size as f32 / 2.0;
        let radius = size as f32 / 3.0;
        
        match shape {
            SignShape::Circle => {
                for i in 0..size {
                    for j in 0..size {
                        let x = i as f32 - center;
                        let y = j as f32 - center;
                        if x.powi(2) + y.powi(2) <= radius.powi(2) {
                            template[[i, j]] = 1.0;
                        }
                    }
                }
            }
            SignShape::Triangle => {
                for i in 0..size {
                    for j in 0..size {
                        if i as f32 >= center && (j as f32 - center).abs() <= (i as f32 - center) {
                            template[[i, j]] = 1.0;
                        }
                    }
                }
            }
            SignShape::Rectangle => {
                let start = (size / 3) as usize;
                let end = (2 * size / 3) as usize;
                for i in start..end {
                    for j in start..end {
                        template[[i, j]] = 1.0;
                    }
                }
            }
            SignShape::Square => {
                let start = (size / 3) as usize;
                let end = (2 * size / 3) as usize;
                for i in start..end {
                    for j in start..end {
                        template[[i, j]] = 1.0;
                    }
                }
            }
            SignShape::Octagon => {
                for i in 0..size {
                    for j in 0..size {
                        let x = (i as f32 - center).abs();
                        let y = (j as f32 - center).abs();
                        if x + y <= radius * 1.5 {
                            template[[i, j]] = 1.0;
                        }
                    }
                }
            }
            SignShape::Diamond => {
                for i in 0..size {
                    for j in 0..size {
                        if (i as f32 - center).abs() + (j as f32 - center).abs() <= radius {
                            template[[i, j]] = 1.0;
                        }
                    }
                }
            }
            SignShape::Pentagon => {
                let angles = [0.0, 72.0, 144.0, 216.0, 288.0];
                let mut points = Vec::new();
                
                for &angle in &angles {
                    let rad = angle * PI / 180.0;
                    let x = center + radius * rad.cos();
                    let y = center + radius * rad.sin();
                    points.push((x, y));
                }
                
                for i in 0..size {
                    for j in 0..size {
                        if Self::point_in_polygon(i as f32, j as f32, &points) {
                            template[[i, j]] = 1.0;
                        }
                    }
                }
            }
        }
        
        template
    }
    
    fn point_in_polygon(x: f32, y: f32, polygon: &[(f32, f32)]) -> bool {
        let mut inside = false;
        let n = polygon.len();
        
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = polygon[i];
            let (xj, yj) = polygon[j];
            
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        
        inside
    }
    
    pub fn process(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((h, w, self.shape_templates.len() + 2));
        
        let mut channel = 0;
        
        for (shape, template) in &self.shape_templates {
            let response = Self::correlate_with_template(input, template);
            
            for i in 0..h {
                for j in 0..w {
                    output[[i, j, channel]] = response[[i, j]];
                }
            }
            
            channel += 1;
        }
        
        output
    }
    
    fn correlate_with_template(input: &Array3<f32>, template: &Array2<f32>) -> Array2<f32> {
        let gray = Self::rgb_to_gray(input);
        let (h, w) = (gray.shape()[0], gray.shape()[1]);
        let (th, tw) = (template.shape()[0], template.shape()[1]);
        let mut response = Array2::zeros((h - th + 1, w - tw + 1));
        
        for i in 0..h - th + 1 {
            for j in 0..w - tw + 1 {
                let mut correlation = 0.0;
                for ti in 0..th {
                    for tj in 0..tw {
                        correlation += gray[[i + ti, j + tj]] * template[[ti, tj]];
                    }
                }
                response[[i, j]] = correlation / (th * tw) as f32;
            }
        }
        
        response
    }
    
    fn rgb_to_gray(rgb: &Array3<f32>) -> Array2<f32> {
        let (h, w, _) = (rgb.shape()[0], rgb.shape()[1], rgb.shape()[2]);
        let mut gray = Array2::zeros((h, w));
        
        for i in 0..h {
            for j in 0..w {
                gray[[i, j]] = 0.299 * rgb[[i, j, 0]]
                    + 0.587 * rgb[[i, j, 1]]
                    + 0.114 * rgb[[i, j, 2]];
            }
        }
        
        gray
    }
}

impl V4ColorProcessing {
    pub fn new() -> Self {
        Self {
            color_channels: Array3::zeros((5, 5, 3)),
            color_constancy: true,
            opponent_colors: OpponentColorSpace {
                red_green: Array2::from_shape_vec((2, 3), vec![1.0, -1.0, 0.0, -1.0, 1.0, 0.0]).unwrap(),
                blue_yellow: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, -1.0, -1.0, 2.0]).unwrap(),
                black_white: Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]).unwrap(),
            },
        }
    }
    
    pub fn process(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((h, w, 6));
        
        for i in 0..h {
            for j in 0..w {
                let r = input[[i, j, 0]];
                let g = input[[i, j, 1]];
                let b = input[[i, j, 2]];
                
                let rg = self.opponent_colors.red_green[[0, 0]] * r
                    + self.opponent_colors.red_green[[0, 1]] * g
                    + self.opponent_colors.red_green[[0, 2]] * b;
                
                let gr = self.opponent_colors.red_green[[1, 0]] * r
                    + self.opponent_colors.red_green[[1, 1]] * g
                    + self.opponent_colors.red_green[[1, 2]] * b;
                
                let by = self.opponent_colors.blue_yellow[[0, 0]] * r
                    + self.opponent_colors.blue_yellow[[0, 1]] * g
                    + self.opponent_colors.blue_yellow[[0, 2]] * b;
                
                let yb = self.opponent_colors.blue_yellow[[1, 0]] * r
                    + self.opponent_colors.blue_yellow[[1, 1]] * g
                    + self.opponent_colors.blue_yellow[[1, 2]] * b;
                
                let bw = self.opponent_colors.black_white[[0, 0]] * r
                    + self.opponent_colors.black_white[[0, 1]] * g
                    + self.opponent_colors.black_white[[0, 2]] * b;
                
                let wb = self.opponent_colors.black_white[[1, 0]] * r
                    + self.opponent_colors.black_white[[1, 1]] * g
                    + self.opponent_colors.black_white[[1, 2]] * b;
                
                output[[i, j, 0]] = rg.max(0.0);
                output[[i, j, 1]] = gr.max(0.0);
                output[[i, j, 2]] = by.max(0.0);
                output[[i, j, 3]] = yb.max(0.0);
                output[[i, j, 4]] = bw.max(0.0);
                output[[i, j, 5]] = wb.max(0.0);
            }
        }
        
        if self.color_constancy {
            output = self.apply_color_constancy(&output);
        }
        
        output
    }
    
    fn apply_color_constancy(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = input.clone();
        
        for ch in 0..c {
            let mut sum = 0.0;
            let mut count = 0;
            
            for i in 0..h {
                for j in 0..w {
                    sum += input[[i, j, ch]];
                    count += 1;
                }
            }
            
            let mean = if count > 0 { sum / count as f32 } else { 0.0 };
            
            if mean > 0.0 {
                let scale = 0.5 / mean;
                
                for i in 0..h {
                    for j in 0..w {
                        output[[i, j, ch]] = (input[[i, j, ch]] * scale).min(1.0);
                    }
                }
            }
        }
        
        output
    }
}

impl ITObjectRecognition {
    pub fn new() -> Self {
        Self {
            feature_hierarchy: FeatureHierarchy {
                low_level: vec!["edges".to_string(), "corners".to_string(), "textures".to_string()],
                mid_level: vec!["shapes".to_string(), "colors".to_string(), "patterns".to_string()],
                high_level: vec!["objects".to_string(), "scenes".to_string(), "context".to_string()],
            },
            invariant_features: Array1::zeros(FEATURE_VECTOR_SIZE),
            object_prototypes: HashMap::new(),
        }
    }
    
    pub fn process(&self, input: &Array3<f32>) -> Array3<f32> {
        let (h, w, c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((h / 4, w / 4, FEATURE_VECTOR_SIZE / 16));
        
        let feature_vector = Self::extract_invariant_features(input);
        
        for i in 0..h / 4 {
            for j in 0..w / 4 {
                for k in 0..FEATURE_VECTOR_SIZE / 16 {
                    let idx = (i * (w / 4) * (FEATURE_VECTOR_SIZE / 16)
                        + j * (FEATURE_VECTOR_SIZE / 16)
                        + k)
                        .min(feature_vector.len() - 1);
                    output[[i, j, k]] = feature_vector[idx];
                }
            }
        }
        
        output
    }
    
    fn extract_invariant_features(input: &Array3<f32>) -> Array1<f32> {
        let (h, w, c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut features = Array1::zeros(FEATURE_VECTOR_SIZE);
        
        let mut idx = 0;
        
        for ch in 0..c {
            let channel = input.slice(s![.., .., ch]);
            let mean = channel.mean().unwrap_or(0.0);
            let std = channel.std(0.0);
            
            if idx < FEATURE_VECTOR_SIZE {
                features[idx] = mean;
                idx += 1;
            }
            
            if idx < FEATURE_VECTOR_SIZE {
                features[idx] = std;
                idx += 1;
            }
        }
        
        let gray = Self::rgb_to_gray(input);
        let gradient_magnitude = Self::compute_gradient_magnitude(&gray);
        let gradient_mean = gradient_magnitude.mean().unwrap_or(0.0);
        let gradient_std = gradient_magnitude.std(0.0);
        
        if idx < FEATURE_VECTOR_SIZE {
            features[idx] = gradient_mean;
            idx += 1;
        }
        
        if idx < FEATURE_VECTOR_SIZE {
            features[idx] = gradient_std;
            idx += 1;
        }
        
        let color_histogram = Self::compute_color_histogram(input);
        for val in color_histogram {
            if idx < FEATURE_VECTOR_SIZE {
                features[idx] = val;
                idx += 1;
            }
        }
        
        features
    }
    
    fn compute_gradient_magnitude(gray: &Array2<f32>) -> Array2<f32> {
        let (h, w) = (gray.shape()[0], gray.shape()[1]);
        let mut gradient = Array2::zeros((h - 1, w - 1));
        
        for i in 0..h - 1 {
            for j in 0..w - 1 {
                let dx = gray[[i + 1, j]] - gray[[i, j]];
                let dy = gray[[i, j + 1]] - gray[[i, j]];
                gradient[[i, j]] = (dx * dx + dy * dy).sqrt();
            }
        }
        
        gradient
    }
    
    fn compute_color_histogram(rgb: &Array3<f32>) -> Vec<f32> {
        let mut histogram = vec![0.0; 64];
        
        let (h, w, _) = (rgb.shape()[0], rgb.shape()[1], rgb.shape()[2]);
        
        for i in 0..h {
            for j in 0..w {
                let r = (rgb[[i, j, 0]] * 4.0).floor() as usize;
                let g = (rgb[[i, j, 1]] * 4.0).floor() as usize;
                let b = (rgb[[i, j, 2]] * 4.0).floor() as usize;
                
                let idx = r * 16 + g * 4 + b;
                if idx < 64 {
                    histogram[idx] += 1.0;
                }
            }
        }
        
        let total = (h * w) as f32;
        if total > 0.0 {
            for val in &mut histogram {
                *val /= total;
            }
        }
        
        histogram
    }
    
    fn rgb_to_gray(rgb: &Array3<f32>) -> Array2<f32> {
        let (h, w, _) = (rgb.shape()[0], rgb.shape()[1], rgb.shape()[2]);
        let mut gray = Array2::zeros((h, w));
        
        for i in 0..h {
            for j in 0..w {
                gray[[i, j]] = 0.299 * rgb[[i, j, 0]]
                    + 0.587 * rgb[[i, j, 1]]
                    + 0.114 * rgb[[i, j, 2]];
            }
        }
        
        gray
    }
}

impl AttentionEngine {
    pub fn new() -> Self {
        Self {
            saliency_maps: Array2::zeros((100, 100)),
            spatial_attention: SpatialAttention {
                focus_map: Array2::zeros((100, 100)),
                inhibition_return: true,
                spotlight_size: 0.2,
            },
            feature_attention: FeatureAttention {
                feature_weights: Array1::ones(FEATURE_VECTOR_SIZE),
                channel_attention: Array1::ones(64),
            },
            temporal_attention: TemporalAttention {
                temporal_filter: Array1::ones(10),
                decay_rate: 0.9,
            },
        }
    }
    
    pub fn compute(&self, input: &Array3<f32>) -> Array2<f32> {
        let (h, w, c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut saliency = Array2::zeros((h, w));
        
        for ch in 0..c {
            let channel = input.slice(s![.., .., ch]);
            let channel_mean = channel.mean().unwrap_or(0.0);
            let channel_std = channel.std(0.0);
            
            let weight = self.feature_attention.channel_attention
                .get(ch)
                .copied()
                .unwrap_or(1.0);
            
            for i in 0..h {
                for j in 0..w {
                    let contrast = (input[[i, j, ch]] - channel_mean).abs();
                    saliency[[i, j]] += contrast * weight;
                }
            }
        }
        
        self.apply_spatial_attention(&saliency)
    }
    
    fn apply_spatial_attention(&self, saliency: &Array2<f32>) -> Array2<f32> {
        let (h, w) = (saliency.shape()[0], saliency.shape()[1]);
        let mut attention = saliency.clone();
        
        if self.spatial_attention.inhibition_return {
            let inhibition_radius = ((h as f32).min(w as f32) * self.spatial_attention.spotlight_size) as usize;
            
            let mut max_val = 0.0;
            let mut max_i = 0;
            let mut max_j = 0;
            
            for i in 0..h {
                for j in 0..w {
                    if attention[[i, j]] > max_val {
                        max_val = attention[[i, j]];
                        max_i = i;
                        max_j = j;
                    }
                }
            }
            
            for i in 0..h {
                for j in 0..w {
                    let distance = ((i as isize - max_i as isize).pow(2)
                        + (j as isize - max_j as isize).pow(2)) as f32
                        .sqrt();
                    
                    if distance < inhibition_radius as f32 {
                        attention[[i, j]] *= 1.5;
                    } else {
                        attention[[i, j]] *= 0.5;
                    }
                }
            }
        }
        
        attention
    }
}

impl MemoryNetwork {
    pub fn new() -> Self {
        Self {
            working_memory: WorkingMemory {
                capacity: 100,
                items: VecDeque::new(),
                decay: 0.95,
            },
            episodic_memory: EpisodicMemory {
                episodes: Vec::new(),
                consolidation_rate: 0.1,
            },
            semantic_memory: SemanticMemory {
                concepts: HashMap::new(),
                associations: HashMap::new(),
            },
        }
    }
}

impl NeuroClassifier {
    pub fn new(num_classes: usize) -> Self {
        Self {
            spiking_network: SpikingNeuralNetwork::new(num_classes),
            decision_layers: Self::create_decision_layers(num_classes),
            confidence_calibration: ConfidenceCalibration {
                temperature: 1.0,
                bins: 10,
                histogram: Array1::zeros(10),
            },
        }
    }
    
    fn create_decision_layers(num_classes: usize) -> Vec<DecisionLayer> {
        let mut layers = Vec::new();
        
        layers.push(DecisionLayer {
            weights: Array2::from_shape_fn((FEATURE_VECTOR_SIZE, 256), |_| {
                rand::random::<f32>() * 0.1
            }),
            biases: Array1::zeros(256),
            activation: ActivationFunction::ReLU,
            dropout: true,
        });
        
        layers.push(DecisionLayer {
            weights: Array2::from_shape_fn((256, 128), |_| {
                rand::random::<f32>() * 0.1
            }),
            biases: Array1::zeros(128),
            activation: ActivationFunction::ReLU,
            dropout: true,
        });
        
        layers.push(DecisionLayer {
            weights: Array2::from_shape_fn((128, num_classes), |_| {
                rand::random::<f32>() * 0.1
            }),
            biases: Array1::zeros(num_classes),
            activation: ActivationFunction::Softplus,
            dropout: false,
        });
        
        layers
    }
    
    pub fn train(
        &mut self,
        features: &[Array1<f32>],
        labels: &[u32],
        learning_rate: f32,
        momentum: f32,
    ) -> TrainingResult {
        let start_time = std::time::Instant::now();
        let num_samples = features.len();
        
        let mut total_loss = 0.0;
        let mut correct = 0;
        
        for epoch in 0..EPOCHS {
            let mut epoch_loss = 0.0;
            let mut epoch_correct = 0;
            
            for batch_start in (0..num_samples).step_by(MINI_BATCH_SIZE) {
                let batch_end = (batch_start + MINI_BATCH_SIZE).min(num_samples);
                let batch_size = batch_end - batch_start;
                
                let batch_features = &features[batch_start..batch_end];
                let batch_labels = &labels[batch_start..batch_end];
                
                let (batch_loss, batch_correct) = self.train_batch(
                    batch_features,
                    batch_labels,
                    learning_rate,
                    momentum,
                );
                
                epoch_loss += batch_loss;
                epoch_correct += batch_correct;
            }
            
            total_loss += epoch_loss;
            correct += epoch_correct;
            
            if epoch % 10 == 0 {
                let accuracy = epoch_correct as f32 / num_samples as f32;
                println!(
                    "Epoch {}: Loss = {:.4}, Accuracy = {:.2}%",
                    epoch,
                    epoch_loss / num_samples as f32,
                    accuracy * 100.0
                );
            }
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        let accuracy = correct as f32 / (num_samples * EPOCHS) as f32;
        let avg_loss = total_loss / (num_samples * EPOCHS) as f32;
        
        TrainingResult {
            accuracy,
            loss: avg_loss,
            training_time,
        }
    }
    
    fn train_batch(
        &mut self,
        features: &[Array1<f32>],
        labels: &[u32],
        learning_rate: f32,
        momentum: f32,
    ) -> (f32, usize) {
        let mut total_loss = 0.0;
        let mut correct = 0;
        
        for (feature, &label) in features.iter().zip(labels.iter()) {
            let prediction = self.forward(feature);
            let loss = self.compute_loss(&prediction, label);
            
            total_loss += loss;
            
            let predicted_class = self.predict_class(&prediction);
            if predicted_class == label {
                correct += 1;
            }
            
            self.backward(feature, &prediction, label, learning_rate, momentum);
        }
        
        (total_loss, correct)
    }
    
    fn forward(&self, features: &Array1<f32>) -> Array1<f32> {
        let mut activations = features.clone();
        
        for layer in &self.decision_layers {
            activations = self.apply_layer(&activations, layer);
        }
        
        activations
    }
    
    fn apply_layer(&self, input: &Array1<f32>, layer: &DecisionLayer) -> Array1<f32> {
        let output_size = layer.weights.shape()[1];
        let mut output = Array1::zeros(output_size);
        
        for j in 0..output_size {
            let mut sum = layer.biases[j];
            
            for i in 0..input.len() {
                sum += input[i] * layer.weights[[i, j]];
            }
            
            output[j] = match layer.activation {
                ActivationFunction::ReLU => sum.max(0.0),
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                ActivationFunction::Tanh => sum.tanh(),
                ActivationFunction::Softplus => (1.0 + sum.exp()).ln(),
                ActivationFunction::Spiking => {
                    if sum > 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
        }
        
        output
    }
    
    fn compute_loss(&self, prediction: &Array1<f32>, true_label: u32) -> f32 {
        let mut max_pred = 0.0;
        for &val in prediction.iter() {
            if val > max_pred {
                max_pred = val;
            }
        }
        
        let mut exp_sum = 0.0;
        for &val in prediction.iter() {
            exp_sum += (val - max_pred).exp();
        }
        
        let log_softmax = prediction[true_label as usize] - max_pred - exp_sum.ln();
        -log_softmax
    }
    
    fn predict_class(&self, prediction: &Array1<f32>) -> u32 {
        let mut max_val = 0.0;
        let mut max_idx = 0;
        
        for (i, &val) in prediction.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        max_idx as u32
    }
    
    fn backward(
        &mut self,
        features: &Array1<f32>,
        prediction: &Array1<f32>,
        true_label: u32,
        learning_rate: f32,
        momentum: f32,
    ) {
    }
    
    pub fn classify(&self, features: &Array1<f32>) -> ClassificationResult {
        let prediction = self.forward(features);
        let predicted_class = self.predict_class(&prediction);
        let confidence = prediction[predicted_class as usize];
        
        ClassificationResult {
            sign_id: predicted_class,
            confidence,
            probabilities: prediction,
        }
    }
}

impl SignDatabase {
    pub fn new() -> Self {
        Self {
            signs: HashMap::new(),
            features: HashMap::new(),
            embeddings: HashMap::new(),
            relationships: HashMap::new(),
        }
    }
}

impl SpikingNeuralNetwork {
    pub fn new(num_classes: usize) -> Self {
        let num_neurons = 1000;
        let mut neurons = Vec::with_capacity(num_neurons);
        
        for _ in 0..num_neurons {
            neurons.push(LIFNeuron {
                membrane_potential: 0.0,
                threshold: 1.0,
                reset_potential: 0.0,
                tau_m: 20.0,
                tau_s: 5.0,
                refractory_period: 5,
            });
        }
        
        let mut synapses = Vec::new();
        for i in 0..num_neurons {
            for j in 0..num_neurons {
                if i != j && rand::random::<f32>() < 0.1 {
                    synapses.push(Synapse {
                        weight: rand::random::<f32>() * 0.2 - 0.1,
                        delay: 1,
                        plasticity: SynapticPlasticity {
                            stdp_rate: 0.01,
                            ltp: 0.1,
                            ltd: 0.1,
                            homeostatic_scaling: true,
                        },
                    });
                }
            }
        }
        
        let layers = vec![
            SNNLayer {
                neurons: neurons[..500].to_vec(),
                connectivity: Array2::zeros((500, 500)),
            },
            SNNLayer {
                neurons: neurons[500..].to_vec(),
                connectivity: Array2::zeros((500, 500)),
            },
        ];
        
        Self {
            neurons,
            synapses,
            layers,
        }
    }
}

fn main() {
    println!("🚀 Khởi động hệ thống nhận diện 200 biển báo giao thông Việt Nam...");
    
    let config = SystemConfig {
        min_detection_confidence: 0.7,
        min_classification_confidence: 0.8,
        max_detections_per_image: 10,
        image_size: (640, 480),
        enable_augmentation: true,
        learning_rate: 0.001,
        momentum: 0.9,
        weight_decay: 0.0001,
        dropout_rate: 0.3,
    };
    
    let mut system = NeuromorphicVisionSystem::new(config);
    
    println!("📂 Đang tải cơ sở dữ liệu biển báo Việt Nam...");
    
    let database_path = Path::new("./vietnamese_traffic_signs");
    match system.load_vietnamese_signs(database_path) {
        Ok(count) => println!("✅ Đã tải {} biển báo", count),
        Err(e) => println!("❌ Lỗi khi tải biển báo: {}", e),
    }
    
    println!("🎯 Hệ thống sẵn sàng nhận diện 200 biển báo!");
    println!("📊 Thông số hệ thống:");
    println!("   • Tổng số lớp biển báo: {}", TOTAL_SIGN_CLASSES);
    println!("   • Kích thước vector đặc trưng: {}", FEATURE_VECTOR_SIZE);
    println!("   • Ngưỡng tin cậy phát hiện: {}", system.config.min_detection_confidence);
    println!("   • Ngưỡng tin cậy phân loại: {}", system.config.min_classification_confidence);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sign_encoding() {
        let system = NeuromorphicVisionSystem::new(SystemConfig::default());
        
        let shape_code = system.encode_shape(&SignShape::Circle);
        assert_eq!(shape_code.len(), 7);
        assert_eq!(shape_code[0], 1.0);
        
        let color_code = system.encode_color(&SignColor::Red);
        assert_eq!(color_code.len(), 9);
        assert_eq!(color_code[0], 1.0);
        
        let category_code = system.encode_category(&SignCategory::Warning);
        assert_eq!(category_code.len(), 6);
        assert_eq!(category_code[0], 1.0);
    }
    
    #[test]
    fn test_retina_processing() {
        let retina = RetinaProcessor::new();
        
        let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
            image::Rgb([(x % 255) as u8, (y % 255) as u8, 128])
        });
        
        let dynamic_img = image::DynamicImage::ImageRgb8(img);
        let output = retina.process(&dynamic_img);
        
        assert_eq!(output.shape().len(), 3);
        assert!(output.shape()[0] > 0);
        assert!(output.shape()[1] > 0);
        assert_eq!(output.shape()[2], 3);
    }
    
    #[test]
    fn test_shape_template_creation() {
        let circle = V2ComplexCells::create_shape_template(&SignShape::Circle, 15);
        assert_eq!(circle.shape(), &[15, 15]);
        
        let triangle = V2ComplexCells::create_shape_template(&SignShape::Triangle, 15);
        assert_eq!(triangle.shape(), &[15, 15]);
        
        let octagon = V2ComplexCells::create_shape_template(&SignShape::Octagon, 15);
        assert_eq!(octagon.shape(), &[15, 15]);
    }
}
