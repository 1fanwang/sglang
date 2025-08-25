import json
import unittest
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from sglang import Engine
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

TEST_IMAGE_URL = "/home/jobuser/sglang/test/lang/example_image.png"


class VLMInputTestBase:
    model_path = None
    chat_template = None
    processor = None
    visual = None  # Should be a callable for precomputed embeddings

    @classmethod
    def setUpClass(cls):
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"
        cls.image_url = TEST_IMAGE_URL
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.main_image = Image.open(cls.image_url)
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
        cls._init_visual()

    @classmethod
    def _init_visual(cls):
        """Override in subclass to set up cls.visual as a callable for precomputed embeddings."""
        raise NotImplementedError

    def setUp(self):
        self.engine = Engine(
            model_path=self.model_path,
            chat_template=self.chat_template,
            device=self.device.type,
            mem_fraction_static=0.8,
            enable_multimodal=True,
            disable_cuda_graph=True,
            trust_remote_code=True,
        )

    def tearDown(self):
        self.engine.shutdown()

    def verify_response(self, output):
        out_text = output["text"].lower()
        assert "taxi" in out_text or "cab" in out_text or "car" in out_text, out_text

    def get_completion_request(self) -> ChatCompletionRequest:
        json_structure = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {"type": "text", "text": "What's in this picture?"},
                    ],
                }
            ],
        }
        json_str = json.dumps(json_structure)
        return ChatCompletionRequest.model_validate_json(json_str)

    def get_processor_output(self, req: Optional[ChatCompletionRequest] = None):
        if req is None:
            req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()

        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[self.main_image],
            return_tensors="pt",
        ).to(self.device)

        return inputs

    async def test_understands_image(self):
        req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()
        output = await self.engine.async_generate(
            prompt=text,
            image_data=[self.main_image],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    async def test_understands_precomputed_embeddings(self):
        req = self.get_completion_request()
        processor_output = self.get_processor_output(req=req)
        with torch.inference_mode():
            precomputed_embeddings = self.__class__.visual(processor_output)
        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
            image_data=[
                self._precomputed_image_data(processor_output, precomputed_embeddings)
            ],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    async def test_understands_pixel_values(self):
        req = self.get_completion_request()
        processor_output = self.get_processor_output(req=req)
        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
            image_data=[self._pixel_values_image_data(processor_output)],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    def _precomputed_image_data(self, processor_output, precomputed_embeddings):
        """This should not be overridden."""
        return dict(
            modality="IMAGE",
            precomputed_embeddings=precomputed_embeddings,
        )

    def _pixel_values_image_data(self, processor_output):
        """Override in subclass to pass the correct set of arguments."""
        raise NotImplementedError


class TestQwenVLUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"

    @classmethod
    def _init_visual(cls):
        cls.visual_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16
            )
            .eval()
            .visual.to(cls.device)
        )
        cls.visual = lambda processor_output: cls.visual_model(
            processor_output["pixel_values"], processor_output["image_grid_thw"]
        )

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            image_grid_thw=processor_output["image_grid_thw"],
            pixel_values=processor_output["pixel_values"],
        )


class TestGemmaUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "google/gemma-3-4b-it"
    chat_template = "gemma-it"

    @classmethod
    def _init_visual(cls):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16
        )
        cls.vision_tower = model.vision_tower.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)
        cls.visual = lambda processor_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=processor_output["pixel_values"]
            ).last_hidden_state
        )

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"][0],
        )


class TestKimiVLImageUnderstandsImage(
    VLMInputTestBase, unittest.IsolatedAsyncioTestCase
):
    model_path = "moonshotai/Kimi-VL-A3B-Instruct"
    chat_template = "kimi-vl"

    @classmethod
    def _init_visual(cls):
        model = AutoModel.from_pretrained(cls.model_path, trust_remote_code=True)
        cls.vision_tower = model.vision_tower.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

        cls.visual = lambda tokenizer_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=tokenizer_output["pixel_values"],
                grid_hws=tokenizer_output["image_grid_hws"],
            )
        )

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"],
            image_grid_hws=processor_output["image_grid_hws"],
        )


class TestInternVLUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "OpenGVLab/InternVL2_5-2B"
    chat_template = "internvl-2-5"

    @classmethod
    def _init_visual(cls):
        # InternVL doesn't use separate pixel_values processing like other VLMs
        # Instead, we need to manually process images using the vision components
        model = AutoModel.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        cls.vision_model = model.vision_model.eval().to(cls.device)
        cls.mlp1 = model.mlp1.eval().to(cls.device)
        
        # For InternVL, we need to manually process the PIL image
        def visual_func(processor_output):
            # Since InternVL processor doesn't return pixel_values, 
            # we need to manually process the image
            # This is a fallback - normally precomputed features would come from external processing
            from transformers.image_utils import make_list_of_images
            from transformers.image_transforms import to_numpy_array, normalize
            import numpy as np
            
            # Use the raw image since processor doesn't give us pixel_values
            image = cls.main_image
            
            # Manual image preprocessing similar to InternVL's preprocessing
            image_array = to_numpy_array(image)
            if image_array.dtype != np.float32:
                image_array = image_array.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            pixel_values = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(cls.device)
            
            # Apply InternVL's vision processing
            vit_embeds = cls.vision_model(pixel_values)
            return cls.mlp1(vit_embeds)
        
        cls.visual = visual_func

    def _pixel_values_image_data(self, processor_output):
        # InternVL doesn't provide pixel_values through processor
        # Instead, we manually process the image
        from transformers.image_utils import make_list_of_images
        from transformers.image_transforms import to_numpy_array, normalize
        import numpy as np
        
        image = self.main_image
        image_array = to_numpy_array(image)
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32) / 255.0
        
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return dict(
            modality="IMAGE",
            pixel_values=pixel_values,
        )


# not for CI: too large
# class TestLlama4ImageUnderstandsImage(
#     VLMInputTestBase, unittest.IsolatedAsyncioTestCase
# ):
#     model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#     chat_template = "llama_4_vision"

#     def setUp(self):
#         self.engine = Engine(
#             model_path=self.model_path,
#             trust_remote_code=True,
#             chat_template=self.chat_template,
#             enable_multimodal=True,
#             mem_fraction_static=0.8,
#             tp_size=4,
#             attention_backend="fa3",
#             context_length=65536,
#         )

#     @classmethod
#     def _init_visual(cls):
#         model = AutoModel.from_pretrained(cls.model_path, trust_remote_code=True, torch_dtype="auto")
#         cls.vision_tower = model.vision_model.eval().to(cls.device)
#         cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

#         cls.visual = lambda tokenizer_output: cls.mm_projector(
#             cls.vision_tower(
#                 pixel_values=tokenizer_output["pixel_values"],
#             ).last_hidden_state.flatten(0, -2)
#         )

#     def _pixel_values_image_data(self, processor_output):
#         return dict(
#             modality="IMAGE",
#             pixel_values=processor_output["pixel_values"],
#         )


if __name__ == "__main__":
    unittest.main()
