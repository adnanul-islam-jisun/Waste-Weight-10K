"""
LLM Reasoning Module
Objective: Synthesize structured data into concise, insightful, human-readable explanations.

This module:
1. Loads and manages Llama 3.1 8B model
2. Processes formatted prompts
3. Generates natural language explanations
"""

import os
import torch
from typing import Optional, Dict, Any, List
import warnings

from config.explanation_config import get_llm_config


class LLMReasoning:
    """
    LLM Reasoning Module using Llama 3.1 8B.
    
    Generates human-readable explanations from structured prompts.
    Supports both local model inference and Hugging Face API.
    
    Args:
        model_name: Hugging Face model identifier
        device: Device to run model on ('auto', 'cuda', 'cpu')
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        use_api: Use Hugging Face Inference API instead of local model
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        use_api: bool = False,
        api_token: Optional[str] = None
    ):
        config = get_llm_config()
        
        self.model_name = model_name or config['model_name']
        self.device = device if device != 'auto' else self._detect_device()
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_api = use_api or config['use_api']
        self.api_token = api_token or config['api_token']
        
        # Generation parameters
        self.generation_config = {
            'max_new_tokens': config['max_new_tokens'],
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'top_k': config['top_k'],
            'do_sample': config['do_sample'],
            'repetition_penalty': config['repetition_penalty']
        }
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Load model based on configuration
        if self.use_api:
            self._setup_api_client()
        else:
            self._load_local_model()
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _load_local_model(self):
        """Load Llama 3.1 8B model locally."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline
            )
            
            # Check for local model path first
            config = get_llm_config()
            model_path = config.get('model_path')
            
            # Use local path if it exists, otherwise use HuggingFace model name
            if model_path and os.path.exists(model_path):
                load_path = model_path
                print(f"Loading from LOCAL: {load_path}")
            else:
                load_path = self.model_name
                print(f"Loading from HuggingFace: {load_path}")
            
            print(f"  Device: {self.device}")
            print(f"  8-bit Quantization: {self.load_in_8bit}")
            print(f"  4-bit Quantization: {self.load_in_4bit}")
            
            # Configure quantization
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
                'device_map': 'auto' if self.device == 'cuda' else None
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                **model_kwargs
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map='auto' if self.device == 'cuda' else None
            )
            
            print(f"✓ Model loaded successfully")
            self._print_memory_usage()
            
        except ImportError as e:
            warnings.warn(
                f"Failed to import transformers: {e}. "
                "Install with: pip install transformers accelerate bitsandbytes"
            )
            self._setup_fallback()
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}")
            self._setup_fallback()
    
    def _setup_api_client(self):
        """Setup Hugging Face Inference API client."""
        if not self.api_token:
            warnings.warn(
                "No API token provided. Set HUGGINGFACE_API_TOKEN environment variable."
            )
            self._setup_fallback()
            return
        
        try:
            from huggingface_hub import InferenceClient
            
            self.api_client = InferenceClient(
                model=self.model_name,
                token=self.api_token
            )
            print(f"✓ Connected to Hugging Face Inference API")
            print(f"  Model: {self.model_name}")
            
        except ImportError:
            warnings.warn(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
            self._setup_fallback()
        except Exception as e:
            warnings.warn(f"Failed to setup API client: {e}")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback when model loading fails."""
        print("⚠ LLM not available. Using template-based explanations.")
        self.use_fallback = True
    
    def _print_memory_usage(self):
        """Print GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def generate(self, prompt: str) -> str:
        """
        Generate explanation from prompt.
        
        Args:
            prompt: Formatted prompt from PromptGenerator
        
        Returns:
            Generated explanation text
        """
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self._generate_fallback(prompt)
        
        if self.use_api:
            return self._generate_via_api(prompt)
        else:
            return self._generate_local(prompt)
    
    def _generate_local(self, prompt: str) -> str:
        """Generate using local model."""
        if self.pipeline is None:
            return self._generate_fallback(prompt)
        
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.generation_config['max_new_tokens'],
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                top_k=self.generation_config['top_k'],
                do_sample=self.generation_config['do_sample'],
                repetition_penalty=self.generation_config['repetition_penalty'],
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            return generated_text
            
        except Exception as e:
            warnings.warn(f"Generation failed: {e}")
            return self._generate_fallback(prompt)
    
    def _generate_via_api(self, prompt: str) -> str:
        """Generate using Hugging Face Inference API."""
        try:
            response = self.api_client.text_generation(
                prompt,
                max_new_tokens=self.generation_config['max_new_tokens'],
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                top_k=self.generation_config['top_k'],
                do_sample=self.generation_config['do_sample'],
                repetition_penalty=self.generation_config['repetition_penalty']
            )
            
            return self._clean_response(response)
            
        except Exception as e:
            warnings.warn(f"API generation failed: {e}")
            return self._generate_fallback(prompt)
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response."""
        # Remove any trailing special tokens
        for token in ['<|eot_id|>', '<|end_of_text|>', '</s>', '<|endoftext|>']:
            text = text.replace(token, '')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper sentence endings
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _generate_fallback(self, prompt: str) -> str:
        """Generate template-based explanation when LLM is unavailable."""
        # Extract key information from prompt
        lines = prompt.split('\n')
        
        prediction = "unknown"
        actual = "unknown"
        image_contrib = "50%"
        metadata_contrib = "50%"
        error = "unknown"
        
        for line in lines:
            if 'Predicted Weight' in line:
                prediction = line.split(':')[-1].strip()
            elif 'Actual Weight' in line and 'Not available' not in line:
                actual = line.split(':')[-1].strip()
            elif 'Image Contribution' in line:
                image_contrib = line.split(':')[-1].strip()
            elif 'Metadata Contribution' in line:
                metadata_contrib = line.split(':')[-1].strip()
            elif 'Absolute Error' in line:
                error = line.split(':')[-1].strip()
        
        explanation = f"""Based on the multimodal analysis, the model predicted a weight of {prediction}. """
        
        if actual != "unknown":
            explanation += f"The actual weight was {actual}, resulting in an absolute error of {error}. "
        
        explanation += f"""The prediction was influenced by both the image ({image_contrib}) and metadata ({metadata_contrib}). """
        
        # Add interpretation based on contributions
        try:
            img_val = float(image_contrib.replace('%', '')) / 100
            if img_val > 0.6:
                explanation += "The visual features from the image were the primary driver of this prediction, suggesting the animal's appearance strongly informed the weight estimate. "
            elif img_val < 0.4:
                explanation += "The metadata features (such as measurements and category) were more influential than the image in this prediction. "
            else:
                explanation += "Both modalities contributed roughly equally to this prediction, indicating a balanced multimodal analysis. "
        except:
            pass
        
        explanation += "For improved accuracy, ensure high-quality images with good lighting and accurate metadata measurements."
        
        return explanation
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate explanations for multiple prompts.
        
        Args:
            prompts: List of formatted prompts
        
        Returns:
            List of generated explanations
        """
        return [self.generate(prompt) for prompt in prompts]
    
    def update_generation_config(self, **kwargs):
        """Update generation parameters."""
        for key, value in kwargs.items():
            if key in self.generation_config:
                self.generation_config[key] = value
                print(f"Updated {key} = {value}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'quantization': '4-bit' if self.load_in_4bit else ('8-bit' if self.load_in_8bit else 'none'),
            'use_api': self.use_api,
            'generation_config': self.generation_config
        }
        
        if self.model is not None:
            info['model_loaded'] = True
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
        elif hasattr(self, 'api_client'):
            info['api_connected'] = True
        else:
            info['fallback_mode'] = True
        
        return info
    
    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ Model unloaded from memory")


class ExplanationGenerator:
    """
    High-level wrapper combining PostHocAnalyzer, PromptGenerator, and LLMReasoning.
    
    Provides a simple interface for generating explanations from predictions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        llm_config: Optional[Dict] = None
    ):
        from .post_hoc_analyzer import PostHocAnalyzer
        from .prompt_generator import PromptGenerator
        
        self.analyzer = PostHocAnalyzer(model=model, device=device)
        self.prompt_generator = PromptGenerator()
        
        # Initialize LLM with optional custom config
        llm_kwargs = llm_config or {}
        self.llm = LLMReasoning(**llm_kwargs)
    
    def explain(
        self,
        image: torch.Tensor,
        category_idx: torch.Tensor,
        numerical_features: torch.Tensor,
        predicted_weight: float,
        actual_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate complete explanation for a prediction.
        
        Args:
            image: Input image tensor
            category_idx: Category index tensor
            numerical_features: Numerical features tensor
            predicted_weight: Predicted weight in kg
            actual_weight: Actual weight in kg (optional)
        
        Returns:
            Dictionary with metrics and explanation
        """
        # Step 1: Analyze prediction
        metrics = self.analyzer.analyze(
            image=image,
            category_idx=category_idx,
            numerical_features=numerical_features,
            predicted_weight=predicted_weight,
            actual_weight=actual_weight
        )
        
        # Step 2: Generate prompt
        prompt = self.prompt_generator.generate(metrics)
        
        # Step 3: Generate explanation
        explanation = self.llm.generate(prompt)
        
        # Combine results
        result = {
            'metrics': metrics,
            'prompt': prompt,
            'explanation': explanation
        }
        
        return result
    
    def explain_batch(
        self,
        images: torch.Tensor,
        category_indices: torch.Tensor,
        numerical_features: torch.Tensor,
        predicted_weights: List[float],
        actual_weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Generate explanations for a batch of predictions."""
        results = []
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            actual = actual_weights[i] if actual_weights else None
            result = self.explain(
                image=images[i:i+1],
                category_idx=category_indices[i:i+1],
                numerical_features=numerical_features[i:i+1],
                predicted_weight=predicted_weights[i],
                actual_weight=actual
            )
            results.append(result)
        
        return results
