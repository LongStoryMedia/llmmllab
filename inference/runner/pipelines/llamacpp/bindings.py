"""
llama.cpp Python SDK - Zero-overhead direct C bindings with full multimodal support.
Matches CLI performance by using direct memory operations and minimal Python overhead.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import threading
import mmap


# Platform-specific library loading
def load_llama_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Load the llama.cpp shared library."""
    if lib_path:
        return ctypes.CDLL(lib_path)
    
    # Try common library names and locations
    lib_names = []
    if sys.platform == "win32":
        lib_names = ["llama.dll", "libllama.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libllama.dylib", "llama.dylib"]
    else:
        lib_names = ["libllama.so", "llama.so"]
    
    # Search paths
    search_paths = [
        ".",
        "./build/bin",
        "./build",
        os.path.expanduser("~/.local/lib"),
        "/usr/local/lib",
        "/usr/lib",
    ]
    
    for path in search_paths:
        for name in lib_names:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                return ctypes.CDLL(full_path)
    
    raise FileNotFoundError(
        f"llama.cpp shared library not found. Tried: {lib_names}\n"
        "Please build llama.cpp with BUILD_SHARED_LIBS=ON"
    )


# C structure definitions - matching llama.cpp exactly
class llama_model(ctypes.Structure):
    pass

class llama_context(ctypes.Structure):
    pass

class clip_ctx(ctypes.Structure):
    pass

class llava_image_embed(ctypes.Structure):
    pass

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
        ("all_pos_0", ctypes.c_int32),
        ("all_pos_1", ctypes.c_int32),
        ("all_seq_id", ctypes.c_int32),
    ]

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_uint32),
        ("n_threads_batch", ctypes.c_uint32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
    ]

class llama_sampler(ctypes.Structure):
    pass


class LlamaSDK:
    """Main SDK class - singleton that wraps llama.cpp C library."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, lib_path: Optional[str] = None):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, lib_path: Optional[str] = None):
        """Initialize the llama.cpp SDK once."""
        if self._initialized:
            return
        
        self.lib = load_llama_library(lib_path)
        self._setup_functions()
        self.lib.llama_backend_init()
        self._initialized = True
    
    def _setup_functions(self):
        """Setup C function signatures - optimized for zero-copy where possible."""
        lib = self.lib
        
        # Backend
        lib.llama_backend_init.argtypes = []
        lib.llama_backend_init.restype = None
        
        lib.llama_backend_free.argtypes = []
        lib.llama_backend_free.restype = None
        
        # Model loading
        lib.llama_model_default_params.argtypes = []
        lib.llama_model_default_params.restype = llama_model_params
        
        lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
        lib.llama_load_model_from_file.restype = ctypes.POINTER(llama_model)
        
        lib.llama_free_model.argtypes = [ctypes.POINTER(llama_model)]
        lib.llama_free_model.restype = None
        
        # Context
        lib.llama_context_default_params.argtypes = []
        lib.llama_context_default_params.restype = llama_context_params
        
        lib.llama_new_context_with_model.argtypes = [ctypes.POINTER(llama_model), llama_context_params]
        lib.llama_new_context_with_model.restype = ctypes.POINTER(llama_context)
        
        lib.llama_free.argtypes = [ctypes.POINTER(llama_context)]
        lib.llama_free.restype = None
        
        # KV cache management
        lib.llama_kv_cache_clear.argtypes = [ctypes.POINTER(llama_context)]
        lib.llama_kv_cache_clear.restype = None
        
        lib.llama_kv_cache_seq_rm.argtypes = [ctypes.POINTER(llama_context), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        lib.llama_kv_cache_seq_rm.restype = ctypes.c_bool
        
        # Tokenization - direct buffer access
        lib.llama_tokenize.argtypes = [
            ctypes.POINTER(llama_model),
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool
        ]
        lib.llama_tokenize.restype = ctypes.c_int32
        
        lib.llama_token_to_piece.argtypes = [
            ctypes.POINTER(llama_model),
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_bool
        ]
        lib.llama_token_to_piece.restype = ctypes.c_int32
        
        # Batch - direct memory access
        lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        lib.llama_batch_init.restype = llama_batch
        
        lib.llama_batch_free.argtypes = [llama_batch]
        lib.llama_batch_free.restype = None
        
        # Decoding
        lib.llama_decode.argtypes = [ctypes.POINTER(llama_context), llama_batch]
        lib.llama_decode.restype = ctypes.c_int32
        
        # Sampling - direct logits access
        lib.llama_get_logits_ith.argtypes = [ctypes.POINTER(llama_context), ctypes.c_int32]
        lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
        
        lib.llama_sampler_chain_init.argtypes = [llama_context_params]
        lib.llama_sampler_chain_init.restype = ctypes.POINTER(llama_sampler)
        
        lib.llama_sampler_chain_add.argtypes = [ctypes.POINTER(llama_sampler), ctypes.POINTER(llama_sampler)]
        lib.llama_sampler_chain_add.restype = None
        
        lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
        lib.llama_sampler_init_temp.restype = ctypes.POINTER(llama_sampler)
        
        lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
        lib.llama_sampler_init_top_k.restype = ctypes.POINTER(llama_sampler)
        
        lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
        lib.llama_sampler_init_top_p.restype = ctypes.POINTER(llama_sampler)
        
        lib.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
        lib.llama_sampler_init_dist.restype = ctypes.POINTER(llama_sampler)
        
        lib.llama_sampler_sample.argtypes = [ctypes.POINTER(llama_sampler), ctypes.POINTER(llama_context), ctypes.c_int32]
        lib.llama_sampler_sample.restype = ctypes.c_int32
        
        lib.llama_sampler_free.argtypes = [ctypes.POINTER(llama_sampler)]
        lib.llama_sampler_free.restype = None
        
        # Utility
        lib.llama_n_vocab.argtypes = [ctypes.POINTER(llama_model)]
        lib.llama_n_vocab.restype = ctypes.c_int32
        
        lib.llama_token_eos.argtypes = [ctypes.POINTER(llama_model)]
        lib.llama_token_eos.restype = ctypes.c_int32
        
        lib.llama_token_bos.argtypes = [ctypes.POINTER(llama_model)]
        lib.llama_token_bos.restype = ctypes.c_int32
        
        lib.llama_n_ctx.argtypes = [ctypes.POINTER(llama_context)]
        lib.llama_n_ctx.restype = ctypes.c_uint32
        
        # Image/multimodal support (llava)
        try:
            lib.clip_model_load.argtypes = [ctypes.c_char_p, ctypes.c_int32]
            lib.clip_model_load.restype = ctypes.POINTER(clip_ctx)
            
            lib.clip_free.argtypes = [ctypes.POINTER(clip_ctx)]
            lib.clip_free.restype = None
            
            lib.llava_image_embed_make_with_filename.argtypes = [
                ctypes.POINTER(clip_ctx),
                ctypes.c_int32,
                ctypes.c_char_p
            ]
            lib.llava_image_embed_make_with_filename.restype = ctypes.POINTER(llava_image_embed)
            
            lib.llava_image_embed_free.argtypes = [ctypes.POINTER(llava_image_embed)]
            lib.llava_image_embed_free.restype = None
            
            lib.llava_eval_image_embed.argtypes = [
                ctypes.POINTER(llama_context),
                ctypes.POINTER(llava_image_embed),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32)
            ]
            lib.llava_eval_image_embed.restype = ctypes.c_bool
            
            self.has_multimodal = True
        except AttributeError:
            self.has_multimodal = False
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'lib') and hasattr(self, '_initialized') and self._initialized:
            self.lib.llama_backend_free()


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_path: str
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    use_mmap: bool = True
    use_mlock: bool = False
    vocab_only: bool = False
    # Multimodal
    clip_model_path: Optional[str] = None  # For vision models (e.g., mmproj file)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    n_ctx: int = 2048
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: Optional[int] = None
    seed: int = 0xFFFFFFFF  # Random seed
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = -1  # -1 = unlimited
    flash_attn: bool = False
    offload_kqv: bool = True  # Offload KV cache to GPU


class LlamaModel:
    """High-level interface for llama.cpp model with direct memory operations."""
    
    def __init__(self, 
                 model_config: ModelConfig,
                 generation_config: Optional[GenerationConfig] = None,
                 sdk: Optional[LlamaSDK] = None):
        """
        Initialize a Llama model.
        
        Args:
            model_config: Model loading configuration
            generation_config: Generation parameters
            sdk: Optional LlamaSDK instance (uses singleton if not provided)
        """
        self.sdk = sdk or LlamaSDK()
        self.model_config = model_config
        self.gen_config = generation_config or GenerationConfig()
        
        # Load model
        self.model = self._load_model()
        self.context = self._create_context()
        
        # Load CLIP model for multimodal if provided
        self.clip_ctx = None
        if model_config.clip_model_path and self.sdk.has_multimodal:
            self.clip_ctx = self._load_clip_model()
        
        # Get special tokens
        self.token_eos = self.sdk.lib.llama_token_eos(self.model)
        self.token_bos = self.sdk.lib.llama_token_bos(self.model)
        self.n_vocab = self.sdk.lib.llama_n_vocab(self.model)
        
        # Pre-allocate buffers for performance
        self._token_buffer = (ctypes.c_int32 * 4096)()  # Reusable token buffer
        self._piece_buffer = ctypes.create_string_buffer(128)  # Reusable decode buffer
        
        # Thread lock for thread-safe generation
        self._lock = threading.Lock()
        
        # Track if model is loaded
        self._is_loaded = True
    
    def _load_model(self) -> ctypes.POINTER(llama_model):
        """Load the model from file."""
        params = self.sdk.lib.llama_model_default_params()
        params.n_gpu_layers = self.model_config.n_gpu_layers
        params.use_mmap = self.model_config.use_mmap
        params.use_mlock = self.model_config.use_mlock
        params.vocab_only = self.model_config.vocab_only
        
        model_path = self.model_config.model_path.encode('utf-8')
        model = self.sdk.lib.llama_load_model_from_file(model_path, params)
        
        if not model:
            raise RuntimeError(f"Failed to load model: {self.model_config.model_path}")
        
        return model
    
    def _load_clip_model(self) -> Optional[ctypes.POINTER(clip_ctx)]:
        """Load CLIP model for vision."""
        if not self.model_config.clip_model_path:
            return None
        
        clip_path = self.model_config.clip_model_path.encode('utf-8')
        clip_ctx = self.sdk.lib.clip_model_load(clip_path, 1)  # 1 = verbosity
        
        if not clip_ctx:
            raise RuntimeError(f"Failed to load CLIP model: {self.model_config.clip_model_path}")
        
        return clip_ctx
    
    def _create_context(self) -> ctypes.POINTER(llama_context):
        """Create inference context."""
        params = self.sdk.lib.llama_context_default_params()
        params.seed = self.gen_config.seed
        params.n_ctx = self.gen_config.n_ctx
        params.n_batch = self.gen_config.n_batch
        params.n_ubatch = self.gen_config.n_ubatch
        params.n_threads = self.gen_config.n_threads or os.cpu_count() or 4
        params.n_threads_batch = params.n_threads
        params.flash_attn = self.gen_config.flash_attn
        params.offload_kqv = self.gen_config.offload_kqv
        params.logits_all = False  # Only get logits for last token
        params.embeddings = False
        
        ctx = self.sdk.lib.llama_new_context_with_model(self.model, params)
        
        if not ctx:
            raise RuntimeError("Failed to create context")
        
        return ctx
    
    def clear(self):
        """Clear the model and context from memory. Call this to free resources."""
        with self._lock:
            if not self._is_loaded:
                return
            
            # Clear KV cache first
            self.sdk.lib.llama_kv_cache_clear(self.context)
            
            # Free clip model if loaded
            if self.clip_ctx:
                self.sdk.lib.clip_free(self.clip_ctx)
                self.clip_ctx = None
            
            # Free context
            if self.context:
                self.sdk.lib.llama_free(self.context)
                self.context = None
            
            # Free model
            if self.model:
                self.sdk.lib.llama_free_model(self.model)
                self.model = None
            
            self._is_loaded = False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._is_loaded
    
    def tokenize(self, text: str, add_bos: bool = True, special: bool = True) -> List[int]:
        """
        Tokenize text using pre-allocated buffer for performance.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            special: Parse special tokens
            
        Returns:
            List of token IDs
        """
        text_bytes = text.encode('utf-8')
        n_tokens = len(self._token_buffer)
        
        n = self.sdk.lib.llama_tokenize(
            self.model,
            text_bytes,
            len(text_bytes),
            self._token_buffer,
            n_tokens,
            add_bos,
            special
        )
        
        if n < 0:
            # Buffer too small, allocate larger
            n_tokens = -n
            self._token_buffer = (ctypes.c_int32 * n_tokens)()
            n = self.sdk.lib.llama_tokenize(
                self.model,
                text_bytes,
                len(text_bytes),
                self._token_buffer,
                n_tokens,
                add_bos,
                special
            )
        
        return list(self._token_buffer[:n])
    
    def detokenize(self, token: int) -> str:
        """
        Convert token ID to text using pre-allocated buffer.
        
        Args:
            token: Token ID
            
        Returns:
            Decoded text
        """
        n = self.sdk.lib.llama_token_to_piece(
            self.model,
            token,
            self._piece_buffer,
            len(self._piece_buffer),
            0,
            False
        )
        
        if n < 0:
            # Need larger buffer
            self._piece_buffer = ctypes.create_string_buffer(-n)
            n = self.sdk.lib.llama_token_to_piece(
                self.model,
                token,
                self._piece_buffer,
                len(self._piece_buffer),
                0,
                False
            )
        
        return self._piece_buffer.value[:n].decode('utf-8', errors='ignore')
    
    def load_image(self, image_path: str) -> ctypes.POINTER(llava_image_embed):
        """
        Load and encode an image for multimodal models.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding pointer
        """
        if not self.clip_ctx:
            raise RuntimeError("CLIP model not loaded. Provide clip_model_path in ModelConfig.")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_path_bytes = image_path.encode('utf-8')
        image_embed = self.sdk.lib.llava_image_embed_make_with_filename(
            self.clip_ctx,
            self.gen_config.n_threads,
            image_path_bytes
        )
        
        if not image_embed:
            raise RuntimeError(f"Failed to load image: {image_path}")
        
        return image_embed
    
    def generate(self,
                 prompt: str,
                 image_path: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 stream: bool = False,
                 stop_sequences: Optional[List[str]] = None) -> Union[str, Iterator[str]]:
        """
        Generate text from a prompt with optional image input.
        
        Args:
            prompt: Input prompt
            image_path: Optional path to image for multimodal models
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            top_k: Top-k sampling (overrides config)
            top_p: Top-p sampling (overrides config)
            stream: Stream tokens as they're generated
            stop_sequences: Optional list of sequences that stop generation
            
        Returns:
            Generated text or iterator of text chunks
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Create a new instance or reload.")
        
        # Use config defaults if not specified
        max_tokens = max_tokens if max_tokens is not None else self.gen_config.max_tokens
        temperature = temperature if temperature is not None else self.gen_config.temperature
        top_k = top_k if top_k is not None else self.gen_config.top_k
        top_p = top_p if top_p is not None else self.gen_config.top_p
        
        if stream:
            return self._generate_stream(prompt, image_path, max_tokens, temperature, top_k, top_p, stop_sequences)
        else:
            return "".join(self._generate_stream(prompt, image_path, max_tokens, temperature, top_k, top_p, stop_sequences))
    
    def _generate_stream(self,
                        prompt: str,
                        image_path: Optional[str],
                        max_tokens: int,
                        temperature: float,
                        top_k: int,
                        top_p: float,
                        stop_sequences: Optional[List[str]]) -> Iterator[str]:
        """Internal streaming generation with direct memory operations."""
        with self._lock:
            # Clear KV cache for fresh generation
            self.sdk.lib.llama_kv_cache_clear(self.context)
            
            # Load image if provided
            image_embed = None
            n_past = 0
            
            if image_path and self.clip_ctx:
                image_embed = self.load_image(image_path)
                
                # Evaluate image embeddings
                n_past_ptr = ctypes.c_int32(0)
                success = self.sdk.lib.llava_eval_image_embed(
                    self.context,
                    image_embed,
                    self.gen_config.n_batch,
                    ctypes.byref(n_past_ptr)
                )
                
                if not success:
                    if image_embed:
                        self.sdk.lib.llava_image_embed_free(image_embed)
                    raise RuntimeError("Failed to evaluate image embeddings")
                
                n_past = n_past_ptr.value
            
            try:
                # Tokenize prompt
                tokens = self.tokenize(prompt, add_bos=not image_path)  # No BOS if image present
                
                # Create batch with direct memory access
                batch = self.sdk.lib.llama_batch_init(self.gen_config.n_batch, 0, 1)
                
                try:
                    # Process prompt tokens
                    for i, token in enumerate(tokens):
                        batch.token[i] = token
                        batch.pos[i] = n_past + i
                        batch.n_seq_id[i] = 1
                        batch.seq_id[i][0] = 0
                        batch.logits[i] = 0  # Don't compute logits for prompt tokens
                    
                    batch.n_tokens = len(tokens)
                    batch.logits[batch.n_tokens - 1] = 1  # Only last token
                    
                    # Evaluate prompt
                    if self.sdk.lib.llama_decode(self.context, batch) != 0:
                        raise RuntimeError("Failed to evaluate prompt")
                    
                    # Create sampler chain
                    sampler = self.sdk.lib.llama_sampler_chain_init(
                        self.sdk.lib.llama_context_default_params()
                    )
                    
                    # Add samplers in optimal order
                    self.sdk.lib.llama_sampler_chain_add(
                        sampler,
                        self.sdk.lib.llama_sampler_init_top_k(top_k)
                    )
                    self.sdk.lib.llama_sampler_chain_add(
                        sampler,
                        self.sdk.lib.llama_sampler_init_top_p(top_p, 1)
                    )
                    self.sdk.lib.llama_sampler_chain_add(
                        sampler,
                        self.sdk.lib.llama_sampler_init_temp(temperature)
                    )
                    self.sdk.lib.llama_sampler_chain_add(
                        sampler,
                        self.sdk.lib.llama_sampler_init_dist(self.gen_config.seed)
                    )
                    
                    # Generate tokens
                    n_cur = n_past + batch.n_tokens
                    n_decode = 0
                    generated_text = ""
                    
                    while True:
                        # Sample next token directly from logits
                        new_token = self.sdk.lib.llama_sampler_sample(sampler, self.context, n_cur - 1)
                        
                        # Check for EOS
                        if new_token == self.token_eos:
                            break
                        
                        # Decode token
                        piece = self.detokenize(new_token)
                        generated_text += piece
                        
                        # Check stop sequences
                        if stop_sequences:
                            should_stop = False
                            for stop_seq in stop_sequences:
                                if stop_seq in generated_text:
                                    idx = generated_text.index(stop_seq)
                                    remaining = generated_text[len(generated_text) - len(piece):idx]
                                    if remaining:
                                        yield remaining
                                    should_stop = True
                                    break
                            if should_stop:
                                break
                        
                        yield piece
                        
                        # Check max tokens
                        n_decode += 1
                        if max_tokens > 0 and n_decode >= max_tokens:
                            break
                        
                        # Prepare next batch (single token)
                        batch.n_tokens = 1
                        batch.token[0] = new_token
                        batch.pos[0] = n_cur
                        batch.n_seq_id[0] = 1
                        batch.seq_id[0][0] = 0
                        batch.logits[0] = 1
                        
                        n_cur += 1