import os
import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging

from PIL import Image
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLModel,
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from gwm_wiser.models.qwen_video_utils import process_vision_info

logger = logging.getLogger(__name__)

# Constants for configuration
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = "<|endoftext|>"


# Define output structure for embeddings
@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


# Define model class to compute embeddings
class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Extract video features from model
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    # Extract image features from model
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules accessible through properties
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # Forward pass through model with input parameters
    # @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        # Pass inputs through the model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        # Return the model output
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

    def encode_video_to_latent(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple:
        """
        Encode video inputs to get video_embeds and deepstack_video_embeds.

        This extracts the latent video representations before they are scattered
        into the text embeddings and passed through the language model.

        Returns:
            Tuple of (video_embeds, deepstack_video_embeds, encode_kwargs)
            - video_embeds: Video features from visual encoder
            - deepstack_video_embeds: Deep stack visual features for language model
            - encode_kwargs: Dict with additional info needed for embed_video_latent()
        """
        if input_ids is None:
            raise ValueError("input_ids is required for encode_video_to_latent()")
        assert pixel_values_videos is not None, "pixel_values_videos is required"

        # Get text embeddings (needed for device/dtype and masking)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Process videos
        video_embeds, deepstack_video_embeds = self.model.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )

        # Get video mask (needed for embed_video_latent)
        _, video_mask = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        visual_pos_masks = video_mask[..., 0]

        # Store additional info needed for embed_video_latent
        encode_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "video_grid_thw": video_grid_thw,
            "visual_pos_masks": visual_pos_masks,
            "video_mask": video_mask,
        }

        return video_embeds, deepstack_video_embeds, encode_kwargs

    def embed_video_latent(
        self,
        video_embeds: torch.FloatTensor,
        deepstack_video_embeds: Optional[List[torch.Tensor]],
        encode_kwargs: dict,
        **kwargs,
    ) -> Qwen3VLForEmbeddingOutput:
        """
        Complete the embedding from encoded video inputs.

        This is the second step of the embedding process, scattering video embeds
        into text embeddings and passing through the language model.

        Args:
            video_embeds: Video features from encode_video_to_latent()
            deepstack_video_embeds: Deep stack visual features
            encode_kwargs: Dict from encode_video_to_latent() with additional info

        Returns:
            Qwen3VLForEmbeddingOutput with last_hidden_state and attention_mask
        """
        input_ids = encode_kwargs["input_ids"]
        attention_mask = encode_kwargs["attention_mask"]
        inputs_embeds = encode_kwargs["inputs_embeds"]
        video_grid_thw = encode_kwargs["video_grid_thw"]
        visual_pos_masks = encode_kwargs["visual_pos_masks"]
        video_mask = encode_kwargs["video_mask"]

        # Scatter video embeddings into text embeddings
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Compute position IDs (same logic as Qwen3VLModel.forward)
        attention_mask_tensor = attention_mask
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(
                attention_mask_tensor[:, 0], dim1=1, dim2=2
            )
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids,
            None,  # image_grid_thw
            video_grid_thw,
            attention_mask=attention_mask_tensor,
        )

        # Pass through language model
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            cache_position=None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_video_embeds,
            **kwargs,
        )

        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def sample_frames(
    frames: List[Union[str, Image.Image]], max_segments: int
) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [frames[frame_idx] for frame_idx in frame_id_list]
    return sampled_frames


def is_image_path(path: str) -> bool:
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
        ".svg",
    }

    if path.startswith(("http://", "https://")):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path

    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True

    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]

        if isinstance(first_elem, Image.Image):
            return True

        if isinstance(first_elem, str):
            return is_image_path(first_elem)

    return False


# Define preprocessor class for formatting and tokenizing inputs
class Qwen3VLPreprocessor:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs,
    ):
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side="right"
        )

    # Truncate token sequence to a specified max length
    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids

        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(
            1 for token_idx in token_ids if token_idx in special_token_ids
        )
        num_non_special_to_keep = max_length - num_special

        final_token_ids = []
        non_special_kept_count = 0
        # Ensure retention of special tokens while truncating the rest
        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
        return final_token_ids

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[
            Union[
                List[Union[str, List[Union[str, Image.Image]]]],
                str,
                List[Union[str, Image.Image]],
            ]
        ] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith(
                "P"
            ):
                instruction = instruction + "."

        # Initialize conversation with system prompts
        content = []
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": instruction or self.default_instruction}
                ],
            },
            {"role": "user", "content": content},
        ]

        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        # Add text, image, or video content to conversation
        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        # Process each video
        for vid in videos:
            video_content = None
            video_kwargs = {"total_pixels": self.total_pixels}

            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ("file://" + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = (
                    vid if vid.startswith(("http://", "https://")) else "file://" + vid
                )
                video_kwargs = {
                    "fps": fps or self.fps,
                    "max_frames": max_frames or self.max_frames,
                }
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append(
                    {"type": "video", "video": video_content, **video_kwargs}
                )

        # Process each image
        for img in images:
            image_content = None

            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = (
                    img if img.startswith(("http://", "https://")) else "file://" + img
                )
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append(
                    {
                        "type": "image",
                        "image": image_content,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )

        # Process each text
        for txt in texts:
            content.append({"type": "text", "text": txt})

        return conversation

    # Preprocess input conversations for model consumption
    def _preprocess_inputs(
        self, conversations: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "NULL"}]}],
                add_generation_prompt=True,
                tokenize=False,
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs


# Define embedder class for processing inputs and generating embeddings
class Qwen3VLEmbedder(Qwen3VLPreprocessor):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs,
    ):
        super().__init__(
            model_name_or_path,
            max_length,
            min_pixels,
            max_pixels,
            total_pixels,
            fps,
            max_frames,
            default_instruction,
            **kwargs,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> tuple:
        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                video=ele.get("video"),
                instruction=ele.get("instruction"),
                fps=ele.get("fps"),
                max_frames=ele.get("max_frames"),
            )
            for ele in inputs
        ]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {
            k: v.to(self.model.device) for k, v in processed_inputs.items()
        }

        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(
            outputs["last_hidden_state"], outputs["attention_mask"]
        )

        # Normalize the embeddings if specified
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def encode_video_to_latent(
        self, inputs: List[Dict[str, Any]] = None, **processed_inputs
    ) -> tuple:
        """
        Encode video inputs to get video_embeds and deepstack_video_embeds.

        Args:
            inputs: List of input dicts (raw inputs).
            **processed_inputs: Preprocessed tensors (input_ids, pixel_values_videos, etc.)
                                If provided, 'inputs' is ignored for preprocessing.

        Returns:
            Tuple of (video_embeds, deepstack_video_embeds)
        """
        if not processed_inputs:
            if inputs is None:
                raise ValueError("Either inputs or processed_inputs must be provided")

            conversations = [
                self.format_model_input(
                    text=ele.get("text"),
                    image=ele.get("image"),
                    video=ele.get("video"),
                    instruction=ele.get("instruction"),
                    fps=ele.get("fps"),
                    max_frames=ele.get("max_frames"),
                )
                for ele in inputs
            ]

            processed_inputs = self._preprocess_inputs(conversations)
            processed_inputs = {
                k: v.to(self.model.device) for k, v in processed_inputs.items()
            }
        else:
            # Ensure inputs are on device
            processed_inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in processed_inputs.items()
            }

        video_embeds, deepstack_video_embeds, _ = self.model.encode_video_to_latent(
            **processed_inputs
        )

        return video_embeds, deepstack_video_embeds

    @torch.no_grad()
    def embed_video_latent(
        self,
        video_embeds: torch.Tensor,
        deepstack_video_embeds: Optional[List[torch.Tensor]],
        inputs: List[Dict[str, Any]] = None,
        normalize: bool = True,
        **processed_inputs,
    ) -> torch.Tensor:
        """
        Complete the embedding from encoded video inputs.

        Args:
            video_embeds: Video features from encode_video_to_latent()
            deepstack_video_embeds: Deep stack visual features
            inputs: Raw input dicts (if processed_inputs not provided)
            normalize: Whether to normalize embeddings
            **processed_inputs: Preprocessed tensors.
        """
        if not processed_inputs:
            if inputs is None:
                raise ValueError("Either inputs or processed_inputs must be provided")

            # Redo preprocessing to get input_ids, attention_mask, etc.
            conversations = [
                self.format_model_input(
                    text=ele.get("text"),
                    image=ele.get("image"),
                    video=ele.get("video"),
                    instruction=ele.get("instruction"),
                    fps=ele.get("fps"),
                    max_frames=ele.get("max_frames"),
                )
                for ele in inputs
            ]

            processed_inputs = self._preprocess_inputs(conversations)
            processed_inputs = {
                k: v.to(self.model.device) for k, v in processed_inputs.items()
            }
        else:
            processed_inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in processed_inputs.items()
            }

        # Build encode_kwargs (same as in model.encode_video_to_latent but without recomputing video_embeds)
        input_ids = processed_inputs["input_ids"]
        attention_mask = processed_inputs["attention_mask"]
        video_grid_thw = processed_inputs.get("video_grid_thw")

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        _, video_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        visual_pos_masks = video_mask[..., 0]

        encode_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "video_grid_thw": video_grid_thw,
            "visual_pos_masks": visual_pos_masks,
            "video_mask": video_mask,
        }

        outputs = self.model.embed_video_latent(
            video_embeds=video_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
            encode_kwargs=encode_kwargs,
        )
        return outputs

    def pooling_video_latent(self, outputs, normalize=True):
        embeddings = self._pooling_last(
            outputs["last_hidden_state"], outputs["attention_mask"]
        )

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
