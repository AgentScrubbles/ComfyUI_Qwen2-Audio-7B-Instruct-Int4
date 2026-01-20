import os
import torch
import torchaudio
import folder_paths
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


class Qwen2_AQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (["Qwen2-Audio-7B-Instruct-Int4"],),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "ComfyUI_Qwen2-Audio-7B-Instruct-Int4"

    def inference(
        self,
        text,
        model,
        keep_model_loaded,
        seed=-1,  # add seed parameter, default is -1
        audio=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"Sergei6000/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
            )
        if self.model is None:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
            )

        with torch.no_grad():
            if audio is not None:
                # Extract waveform and sample rate from AUDIO dict
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]

                # Remove batch dimension if present: (batch, channels, samples) -> (channels, samples)
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)

                # Convert to mono if stereo: (channels, samples) -> (samples,)
                if waveform.dim() == 2 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                elif waveform.dim() == 2:
                    waveform = waveform.squeeze(0)

                # Resample to processor's expected sample rate if needed
                target_sr = self.processor.feature_extractor.sampling_rate
                if sample_rate != target_sr:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

                # Convert to numpy array (processor expects numpy)
                audio_array = waveform.numpy()

                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": "audio_input"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )

                inputs = self.processor(
                    text=text, audios=[audio_array], return_tensors="pt", padding=True
                )
                inputs = inputs.to("cuda")

                generate_ids = self.model.generate(**inputs, max_length=256)
                generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

            else:
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )

                inputs = self.processor(
                    text=text, return_tensors="pt", padding=True
                )
                inputs = inputs.to("cuda")

                generate_ids = self.model.generate(**inputs, max_length=256)
                generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
                # raise ValueError("Either audio or text must be provided")

            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if not keep_model_loaded:
                del self.processor  # release tokenizer memory
                del self.model  # release model memory
                self.processor = None  # set tokenizer to None
                self.model = None  # set model to None
                torch.cuda.empty_cache()  # release GPU memory
                torch.cuda.ipc_collect()

            return (response,)
