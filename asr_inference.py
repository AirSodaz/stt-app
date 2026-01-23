import os
import threading
import torch
import logging
import gc
from typing import Optional, Dict, List, Any
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
# Pre-import Fun-ASR-Nano model to register it in FunASR's tables
# This is needed because FunASR 1.3.0 doesn't auto-import model modules
try:
    import funasr.models.fun_asr_nano.model  # noqa: F401
except ImportError:
    pass  # Model module may not exist in older versions
from modelscope import snapshot_download
from omegaconf import OmegaConf
import re
import numpy as np

# logging config is handled by the main entry point (server.py)
logger = logging.getLogger(__name__)

# SenseVoice emoji mappings (official)
SENSEVOICE_EMOJI_DICT = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

# Sort keys by length descending to handle multi-tag patterns first
sorted_tags = sorted(SENSEVOICE_EMOJI_DICT.keys(), key=lambda k: -len(k))

# Create a single regex for all tags. This is more efficient than iterating.
SENSEVOICE_TAG_PATTERN = re.compile("|".join(re.escape(tag) for tag in sorted_tags))

# Compiled regex patterns for sensevoice_postprocess optimization
TAG_NORMALIZATION_PATTERN = re.compile(r'<\s*\|\s*([^|]*?)\s*\|\s*>')
UNRECOGNIZED_TAG_PATTERN = re.compile(r'<\|[^|]*\|>')
WHITESPACE_PATTERN = re.compile(r'\s+')


def sensevoice_postprocess(text: str, show_emoji: bool = True) -> str:
    """
    Post-process SenseVoice output text.
    
    Args:
        text: Raw transcription text with tags like <|zh|>, <|HAPPY|>, etc.
        show_emoji: If True, convert emotion/event tags to emojis.
                   If False, remove all tags without showing emojis.
    
    Returns:
        Cleaned transcription text with optional emojis.
    """
    # First normalize tags with spaces: "< | zh | >" -> "<|zh|>"
    text = TAG_NORMALIZATION_PATTERN.sub(lambda m: f'<|{m.group(1).strip()}|>', text)
    

    def replace_tag(match):
        """Replacer function for re.sub."""
        tag = match.group(0)
        if show_emoji:
            return SENSEVOICE_EMOJI_DICT.get(tag, "")
        return "" # If not showing emoji, remove all known tags

    # Use the pre-compiled regex to replace all known tags in one pass
    text = SENSEVOICE_TAG_PATTERN.sub(replace_tag, text)
    
    # Remove any remaining unrecognized tags
    text = UNRECOGNIZED_TAG_PATTERN.sub('', text)
    
    # Clean up extra spaces
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    return text

class ASRModelManager:
    """
    Manager class for handling multiple ASR models using FunASR.
    Supports loading, switching, and transcribing with different models.
    """
    
    # Model definitions
    MODELS = {
        "SenseVoiceSmall": "iic/SenseVoiceSmall",
        "Paraformer": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "Paraformer-Online": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "Fun-ASR-Nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    }

    # Model types (offline vs online)
    MODEL_TYPES = {
        "SenseVoiceSmall": "offline",
        "Paraformer": "offline",
        "Paraformer-Online": "online",
        "Fun-ASR-Nano": "offline",
    }

    def __init__(self, model_cache_dir: str = "./models"):
        """
        Initialize the ASRModelManager.

        Args:
            model_cache_dir (str): Local directory to cache downloaded models.
        """
        self.model_cache_dir = model_cache_dir
        self.device = self._get_device()
        self.current_model_name: Optional[str] = None
        self.model: Optional[AutoModel] = None
        
        # Ensure the model cache directory exists
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Note: We no longer auto-load a model on init.
        # Models are loaded on demand when transcribe() is called.
        # Use RLock to allow re-entrant locking (transcribe calls load_model)
        self.lock = threading.RLock()
        # Use RLock for download_lock to allow re-entrant calls from load_model -> download_model
        self.download_lock = threading.RLock()
        logger.info("ASRModelManager initialized. No model loaded yet.")

    def _get_device(self) -> str:
        """
        Determine the most appropriate device for inference.
        """
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def is_model_downloaded(self, model_name: str) -> bool:
        """
        Check if the model is locally available.
        """
        if model_name not in self.MODELS:
            return False
            
        model_id = self.MODELS[model_name]

        model_path = os.path.join(self.model_cache_dir, model_id)
        return os.path.exists(os.path.join(model_path, "config.yaml"))

    def get_model_type(self, model_name: str) -> str:
        """
        Get the type of the model (offline or online).
        Defaults to 'offline' if unknown.
        """
        return self.MODEL_TYPES.get(model_name, "offline")

    def get_downloaded_models(self) -> List[str]:
        """
        Return list of model names that are downloaded locally.
        """
        return [name for name in self.MODELS.keys() if self.is_model_downloaded(name)]

    def download_model(self, model_name: str) -> str:
        """
        Explicitly download a model.
        """
        if model_name not in self.MODELS:
             raise ValueError(f"Unknown model: {model_name}")
        
        # Ensure only one download happens at a time (globally or per model logic if extended)
        with self.download_lock:
            model_id = self.MODELS[model_name]
            logger.info(f"Downloading model '{model_name}' ({model_id})...")

            try:
                model_dir = snapshot_download(model_id, cache_dir=self.model_cache_dir)
                logger.info(f"Model '{model_name}' downloaded to {model_dir}")
                return model_dir
            except Exception as e:
                logger.error(f"Failed to download model '{model_name}': {e}")
                raise
    
    def get_model_path(self, model_name: str) -> str:
        """Get the expected local path for a model."""
        if model_name not in self.MODELS:
            return None
        model_id = self.MODELS[model_name]
        return os.path.join(self.model_cache_dir, model_id)

    def load_model(self, model_name: str) -> None:
        """
        Load a specific model by name. If the model is already loaded, do nothing.

        Args:
            model_name (str): The specific name of the model to load (must be in self.MODELS).
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(self.MODELS.keys())}")

        if self.current_model_name == model_name and self.model is not None:
             logger.info(f"Model '{model_name}' is already loaded.")
             return

        # 1. Ensure download (OUTSIDE lock)
        if not self.is_model_downloaded(model_name):
            with self.download_lock:
                 # Double-check inside download lock to prevent multiple downloads
                 if not self.is_model_downloaded(model_name):
                     logger.info(f"Model '{model_name}' not found locally. Downloading...")
                     self.download_model(model_name)

        with self.lock:
            # Double-check inside lock
            if self.current_model_name == model_name and self.model is not None:
                return

            logger.info(f"Switching model to '{model_name}'...")

        
            # Unload existing model to free memory
            if self.model is not None:
                del self.model
                self.model = None
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Previous model unloaded.")
            
            try:
                # Get path directly (should be downloaded now)
                model_dir = self.get_model_path(model_name)
                if not model_dir or not os.path.exists(model_dir):
                    # Fallback just in case something went wrong or race condition
                    # This might block, but it shouldn't happen if download succeeded above
                    logger.warning(f"Model path for {model_name} not found despite download check. Re-downloading...")
                    model_dir = self.download_model(model_name)
                
                # 2. Initialize the model
                # Special handling for each model to fix FunASR 1.3.0 vocab_size bug
                if model_name == "SenseVoiceSmall":
                    self.model = self._initialize_sense_voice(model_dir)
                elif model_name == "Paraformer":
                    self.model = self._initialize_paraformer_offline(model_dir)
                elif model_name == "Paraformer-Online":
                    self.model = self._initialize_paraformer_online(model_dir)
                elif model_name == "Fun-ASR-Nano":
                    self.model = self._initialize_fun_asr_nano(model_dir)
                else:
                    self.model = self._initialize_generic(model_dir, model_name)
                    
                self.current_model_name = model_name
                logger.info(f"Model '{model_name}' loaded successfully.")
                
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                self.current_model_name = None # Reset state on failure
                raise

    def _initialize_sense_voice(self, model_dir: str) -> AutoModel:
        """
        Specific initialization for SenseVoiceSmall with config patches.
        """
        config_path = os.path.join(model_dir, "config.yaml")
        config = OmegaConf.load(config_path)

        # Build tokenizer_conf
        tokenizer_conf: Dict[str, str] = {
            "token_list": os.path.join(model_dir, "tokens.json"),
        }
        bpe_model_path = os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        if os.path.exists(bpe_model_path):
            tokenizer_conf["bpemodel"] = bpe_model_path

        # Build frontend_conf
        frontend_conf = dict(config.get("frontend_conf", {}))
        mvn_file_path = os.path.join(model_dir, "am.mvn")
        if os.path.exists(mvn_file_path):
            frontend_conf["cmvn_file"] = mvn_file_path

        return AutoModel(
            model=config["model"],
            model_path=model_dir,
            model_conf=dict(config.get("model_conf", {})),
            init_param=os.path.join(model_dir, "model.pt"),
            tokenizer=config.get("tokenizer", "SentencepiecesTokenizer"),
            tokenizer_conf=tokenizer_conf,
            frontend=config.get("frontend", "WavFrontend"),
            frontend_conf=frontend_conf,
            encoder=config.get("encoder"),
            encoder_conf=dict(config.get("encoder_conf", {})),
            specaug=config.get("specaug"),
            specaug_conf=dict(config.get("specaug_conf", {})),
            trust_remote_code=False,
            device=self.device,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            # punc_model="ct-punc",
            disable_update=True,
        )

    def _initialize_paraformer_offline(self, model_dir: str) -> AutoModel:
        """
        Initialize Paraformer offline model for batch/file transcription.
        This model has built-in VAD and punctuation, optimized for non-realtime use.
        Uses the speech_paraformer-large-vad-punc model.
        """
        config_path = os.path.join(model_dir, "config.yaml")
        config = OmegaConf.load(config_path)

        # Build tokenizer_conf for CharTokenizer
        tokenizer_conf: Dict[str, Any] = {
            "token_list": os.path.join(model_dir, "tokens.json"),
        }
        yaml_tokenizer_conf = config.get("tokenizer_conf", {})
        if yaml_tokenizer_conf:
            tokenizer_conf.update(dict(yaml_tokenizer_conf))

        # Build frontend_conf
        frontend_conf = dict(config.get("frontend_conf", {}))
        mvn_file_path = os.path.join(model_dir, "am.mvn")
        if os.path.exists(mvn_file_path):
            frontend_conf["cmvn_file"] = mvn_file_path

        return AutoModel(
            model=config["model"],
            model_path=model_dir,
            model_conf=dict(config.get("model_conf", {})),
            init_param=os.path.join(model_dir, "model.pt"),
            tokenizer=config.get("tokenizer", "CharTokenizer"),
            tokenizer_conf=tokenizer_conf,
            frontend=config.get("frontend", "WavFrontend"),
            frontend_conf=frontend_conf,
            encoder=config.get("encoder"),
            encoder_conf=dict(config.get("encoder_conf", {})),
            decoder=config.get("decoder"),
            decoder_conf=dict(config.get("decoder_conf", {})),
            predictor=config.get("predictor"),
            predictor_conf=dict(config.get("predictor_conf", {})),
            specaug=config.get("specaug"),
            specaug_conf=dict(config.get("specaug_conf", {})),
            ctc_conf=dict(config.get("ctc_conf", {})),
            trust_remote_code=False,
            device=self.device,
            # Even though the model name suggests built-in VAD/punc, 
            # we still need to add them for proper Chinese text output
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="ct-punc",
            disable_update=True,
        )

    def _initialize_paraformer_online(self, model_dir: str) -> AutoModel:
        """
        Specific initialization for Paraformer streaming model with config patches.
        Works around FunASR 1.3.0 bug by explicitly loading all config parameters.
        NOTE: No VAD/punc for streaming mode - they interfere with chunk-by-chunk inference.
        """
        config_path = os.path.join(model_dir, "config.yaml")
        config = OmegaConf.load(config_path)

        # Build tokenizer_conf for CharTokenizer
        tokenizer_conf: Dict[str, Any] = {
            "token_list": os.path.join(model_dir, "tokens.json"),
        }
        # Add additional tokenizer config from yaml
        yaml_tokenizer_conf = config.get("tokenizer_conf", {})
        if yaml_tokenizer_conf:
            tokenizer_conf.update(dict(yaml_tokenizer_conf))

        # Build frontend_conf for WavFrontendOnline
        frontend_conf = dict(config.get("frontend_conf", {}))
        mvn_file_path = os.path.join(model_dir, "am.mvn")
        if os.path.exists(mvn_file_path):
            frontend_conf["cmvn_file"] = mvn_file_path

        return AutoModel(
            model=config["model"],
            model_path=model_dir,
            model_conf=dict(config.get("model_conf", {})),
            init_param=os.path.join(model_dir, "model.pt"),
            tokenizer=config.get("tokenizer", "CharTokenizer"),
            tokenizer_conf=tokenizer_conf,
            frontend=config.get("frontend", "WavFrontendOnline"),
            frontend_conf=frontend_conf,
            encoder=config.get("encoder"),
            encoder_conf=dict(config.get("encoder_conf", {})),
            decoder=config.get("decoder"),
            decoder_conf=dict(config.get("decoder_conf", {})),
            predictor=config.get("predictor"),
            predictor_conf=dict(config.get("predictor_conf", {})),
            specaug=config.get("specaug"),
            specaug_conf=dict(config.get("specaug_conf", {})),
            ctc_conf=dict(config.get("ctc_conf", {})),
            trust_remote_code=False,
            device=self.device,
            punc_model="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
            vad_kwargs={"max_single_segment_time": 30000},
            disable_update=True,
        )

    def _initialize_fun_asr_nano(self, model_dir: str) -> AutoModel:
        """
        Specific initialization for Fun-ASR-Nano model with config patches.
        Works around FunASR 1.3.0 bug by explicitly loading all config parameters.
        """
        config_path = os.path.join(model_dir, "config.yaml")
        config = OmegaConf.load(config_path)

        # Build tokenizer_conf for HuggingfaceTokenizer
        tokenizer_conf: Dict[str, Any] = {}
        llm_path = os.path.join(model_dir, "Qwen3-0.6B")
        if os.path.exists(llm_path):
            tokenizer_conf["init_param_path"] = llm_path

        # Build frontend_conf for WavFrontend
        frontend_conf = dict(config.get("frontend_conf", {}))
        # Fun-ASR-Nano doesn't use am.mvn but uses multilingual.tiktoken for tokenization

        # Build llm_conf with resolved paths
        llm_conf = dict(config.get("llm_conf", {}))
        # Override relative path with absolute path
        llm_relative_path = llm_conf.get("init_param_path", "Qwen3-0.6B")
        llm_absolute_path = os.path.join(model_dir, llm_relative_path)
        if os.path.exists(llm_absolute_path):
            llm_conf["init_param_path"] = llm_absolute_path

        return AutoModel(
            model=config["model"],
            model_path=model_dir,
            model_conf=dict(config.get("model_conf", {})),
            init_param=os.path.join(model_dir, "model.pt"),
            tokenizer=config.get("tokenizer", "HuggingfaceTokenizer"),
            tokenizer_conf=tokenizer_conf,
            frontend=config.get("frontend", "WavFrontend"),
            frontend_conf=frontend_conf,
            audio_encoder=config.get("audio_encoder"),
            audio_encoder_conf=dict(config.get("audio_encoder_conf", {})),
            llm=config.get("llm"),
            llm_conf=llm_conf,
            audio_adaptor=config.get("audio_adaptor"),
            audio_adaptor_conf=dict(config.get("audio_adaptor_conf", {})),
            ctc_decoder=config.get("ctc_decoder"),
            ctc_decoder_conf=dict(config.get("ctc_decoder_conf", {})),
            ctc_conf=dict(config.get("ctc_conf", {})),
            trust_remote_code=True,  # Fun-ASR-Nano uses custom components
            device=self.device,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            disable_update=True,
        )

    def _initialize_generic(self, model_dir: str, model_name: str) -> AutoModel:
        """
        Generic initialization for other models (Paraformer, Fun-ASR-Nano).
        """
        
        kwargs = {
            "model": model_dir,
            "device": self.device,
            "disable_update": True,
            "trust_remote_code": True, # Some newer models might need this
        }

        # For streaming or frame-level models, they often need VAD to segment long files
        if model_name in ["Paraformer", "Fun-ASR-Nano"]:
             kwargs["vad_model"] = "fsmn-vad"
             kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}
             kwargs["punc_model"] = "ct-punc" # Add punctuation restoration if supported
        
        return AutoModel(**kwargs)

    def transcribe(self, audio_input, language: str = "auto", use_itn: bool = False, model_name: str = None, show_emoji: bool = True) -> str:
        """
        Transcribe audio from file path or numpy array.
        
        Args:
            audio_input: Either a file path (str) or numpy array of audio samples
            language: Language code
            use_itn: Whether to use inverse text normalization
            model_name: Model to use
            show_emoji: Whether to show emojis for emotion/event tags (SenseVoice only)
        """
        
        # Check if it's a file path or numpy array
        is_file = isinstance(audio_input, str)
        
        if is_file and not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        
        # Determine which model to use
        target_model = model_name if model_name else self.current_model_name
        if not target_model:
            raise ValueError("No model specified and no model currently loaded.")
        
        # Ensure it is downloaded AND loaded
        self.load_model(target_model)

        if self.model is None:
            raise RuntimeError("No model loaded.")

        # Acquire lock for thread-safe inference
        with self.lock:
            try:
                # Prepare generate kwargs
                generate_kwargs = {
                    "input": audio_input,
                    "use_itn": use_itn,
                }
                
                # Fun-ASR-Nano does not support batch_size_s (it uses LLM decoding)
                # It requires batch_size=1 and input as a list
                if self.current_model_name == "Fun-ASR-Nano":
                     generate_kwargs["batch_size"] = 1
                     generate_kwargs["input"] = [audio_input]
                     # Fix for "The attention mask and the pad token id were not set" warning
                     # Fun-ASR-Nano uses Qwen tokenizer where eos_token_id is 151645
                     # We explicitly set pad_token_id to eos_token_id on the generation_config to suppress warnings
                     try:
                         if hasattr(self.model, "model") and hasattr(self.model.model, "llm"):
                             llm = self.model.model.llm
                             if hasattr(llm, "generation_config") and llm.generation_config.pad_token_id is None:
                                  llm.generation_config.pad_token_id = llm.generation_config.eos_token_id
                                  logger.info(f"Set Fun-ASR-Nano pad_token_id to {llm.generation_config.pad_token_id}")
                     except Exception as e:
                         logger.warning(f"Failed to set pad_token_id for Fun-ASR-Nano: {e}")
                else:
                     generate_kwargs["batch_size_s"] = 60
                
                # SenseVoice specific params
                if self.current_model_name == "SenseVoiceSmall":
                    generate_kwargs["language"] = language
                    generate_kwargs["merge_vad"] = True
                    generate_kwargs["merge_length_s"] = 15
                
                # Paraformer specific params
                # Note: Do NOT set "cache" manually for Paraformer streaming unless managing state.
                # The model will initialize it if missing.
                # if self.current_model_name == "Paraformer":
                #      generate_kwargs["cache"] = {"start_idx": 0}
                
                # Run inference
                res = self.model.generate(**generate_kwargs)
                
                if not res:
                    return ""

                # Post-process
                text = ""
                if isinstance(res, list) and len(res) > 0:
                    text = res[0].get("text", "")
                
                if self.current_model_name == "SenseVoiceSmall":
                     text = sensevoice_postprocess(text, show_emoji=show_emoji)
                
                return text

            except Exception as e:
                logger.error(f"Error during transcription with {self.current_model_name}: {e}")
                raise

    def inference_chunk(self, audio_chunk, cache: Dict[str, Any] = None, is_final: bool = False, model_name: str = None) -> Any:
        """
        Run inference on a single audio chunk (streaming).
        
        Args:
            audio_chunk: Audio data (bytes or numpy array).
            cache: State dictionary for the streaming model.
            is_final: Whether this is the final chunk.
            model_name: Optional model name.
            
        Returns:
            res: Result from model.generate (usually list of dicts).
        """
        # Determine which model to use
        target_model = model_name if model_name else self.current_model_name
        if not target_model:
            raise ValueError("No model specified and no model currently loaded.")
            
        # Ensure it is downloaded AND loaded
        if self.current_model_name != target_model or self.model is None:
             self.load_model(target_model)

        if self.model is None:
            raise RuntimeError("No model loaded.")
            
        if target_model != "Paraformer":
             # For now, only Paraformer is verified for detailed streaming here
             # But Fun-ASR-Nano might work if supported.
             # raising warning or passing through
             pass

        # Acquire lock for thread-safe inference
        with self.lock:
            try:
                # Convert raw PCM bytes to numpy array
                # Frontend sends Int16 PCM at 16kHz
                if isinstance(audio_chunk, bytes) and len(audio_chunk) > 0:
                    # Convert Int16 bytes to numpy array
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    # Normalize to float32 [-1, 1] as expected by the model
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    logger.debug(f"[Streaming] Received {len(audio_chunk)} bytes -> {len(audio_data)} samples")
                elif isinstance(audio_chunk, bytes) and len(audio_chunk) == 0:
                    audio_data = np.array([], dtype=np.float32)
                    logger.debug("[Streaming] Received EOF (empty bytes)")
                else:
                    audio_data = audio_chunk
                
                generate_kwargs = {
                    "input": audio_data,
                    "cache": cache if cache is not None else {},
                    "is_final": is_final,
                    # "chunk_size": [0, 10, 5], # Removed to avoid conflict with VAD
                    "encoder_chunk_look_back": 4,
                    "decoder_chunk_look_back": 1,
                }

                logger.debug(f"[Streaming] Calling generate with is_final={is_final}")
                res = self.model.generate(**generate_kwargs)
                logger.debug(f"[Streaming] Result: {res}")
                return res
                
            except Exception as e:
                 logger.error(f"Error during streaming inference: {e}")
                 raise


