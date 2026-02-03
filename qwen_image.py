import torch
import numpy as np
from textwrap import dedent

from diffusers import DiffusionPipeline
from diffusers.pipelines.qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm
import time
from json_parsing import get_scene_data_by_group_id, get_goals_data_by_group_id, get_all_group_ids

# ========================
# 0) 커스텀 파이프라인: 3-GPU(1,2,3) 수동 분산
# ========================
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

class CustomQwenImagePipeline(QwenImagePipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        # 고정 디바이스 매핑: 1,2,3번 GPU 사용
        dev_main = "cuda:1"
        dev_blk_a = "cuda:2"  # 앞 블록
        dev_blk_b = "cuda:3"  # 뒷 블록

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1) 입력 검증
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        # 2) 배치 크기 결정
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3) 임베딩 준비
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=dev_main,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=dev_main,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4) 잠복변수 준비
        num_channels_latents = self.transformer.config.in_channels // 4
        prep = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            dev_main,
            generator,
            latents,
        )
        latents = prep[0] if isinstance(prep, (tuple, list)) else prep
        
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5) 스케줄러 타임스텝
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            dev_main,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # guidance 텐서
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=dev_main, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6) 디노이징 루프 (수동 디바이스 라우팅)
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # cond 패스
                with self.transformer.cache_context("cond"):
                    hidden_states = self.transformer.img_in(latents).to(dev_main)
                    encoder_hidden_states = self.transformer.txt_norm(prompt_embeds).to(dev_main)
                    encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                    encoder_hidden_states_mask = prompt_embeds_mask.to(dev_main)

                    timestep_float = timestep.to(dev_main) / 1000
                    temb = (
                        self.transformer.time_text_embed(timestep_float, hidden_states)
                        if guidance is None
                        else self.transformer.time_text_embed(timestep_float, guidance.to(dev_main), hidden_states)
                    )
                    image_rotary_emb = self.transformer.pos_embed(
                        img_shapes,
                        prompt_embeds_mask.sum(dim=1).tolist(),
                        device=dev_main,
                    )

                    # 앞 구간: cuda:2
                    temb_a = temb.to(dev_blk_a, non_blocking=True)
                    image_rotary_emb_a = tuple(item.to(dev_blk_a, non_blocking=True) for item in image_rotary_emb)
                    torch.cuda.synchronize(dev_blk_a)
                    for j, block in enumerate(self.transformer.transformer_blocks[:30]):
                        hidden_states = hidden_states.to(dev_blk_a, non_blocking=True)
                        encoder_hidden_states = encoder_hidden_states.to(dev_blk_a, non_blocking=True)
                        encoder_hidden_states_mask = encoder_hidden_states_mask.to(dev_blk_a, non_blocking=True)
                        torch.cuda.synchronize(dev_blk_a)

                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb_a,
                            image_rotary_emb=image_rotary_emb_a,
                            joint_attention_kwargs=self.attention_kwargs,
                        )

                    # 뒷 구간: cuda:3
                    temb_b = temb.to(dev_blk_b, non_blocking=True)
                    image_rotary_emb_b = tuple(item.to(dev_blk_b, non_blocking=True) for item in image_rotary_emb)
                    torch.cuda.synchronize(dev_blk_b)
                    for j, block in enumerate(self.transformer.transformer_blocks[30:], start=30):
                        hidden_states = hidden_states.to(dev_blk_b, non_blocking=True)
                        encoder_hidden_states = encoder_hidden_states.to(dev_blk_b, non_blocking=True)
                        encoder_hidden_states_mask = encoder_hidden_states_mask.to(dev_blk_b, non_blocking=True)
                        torch.cuda.synchronize(dev_blk_b)

                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb_b,
                            image_rotary_emb=image_rotary_emb_b,
                            joint_attention_kwargs=self.attention_kwargs,
                        )

                    # 출력 정규화 및 예측은 다시 cuda:1
                    hidden_states = hidden_states.to(dev_main, non_blocking=True)
                    torch.cuda.synchronize(dev_main)
                    hidden_states = self.transformer.norm_out(hidden_states, temb.to(dev_main))
                    noise_pred = self.transformer.proj_out(hidden_states)

                # true CFG일 때 uncond 패스
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        hidden_states = self.transformer.img_in(latents).to(dev_main)
                        encoder_hidden_states = self.transformer.txt_norm(negative_prompt_embeds).to(dev_main)
                        encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                        encoder_hidden_states_mask = negative_prompt_embeds_mask.to(dev_main)

                        timestep_float = timestep.to(dev_main) / 1000
                        temb = (
                            self.transformer.time_text_embed(timestep_float, hidden_states)
                            if guidance is None
                            else self.transformer.time_text_embed(timestep_float, guidance.to(dev_main), hidden_states)
                        )
                        image_rotary_emb = self.transformer.pos_embed(
                            img_shapes,
                            negative_prompt_embeds_mask.sum(dim=1).tolist(),
                            device=dev_main,
                        )

                        temb_a = temb.to(dev_blk_a, non_blocking=True)
                        image_rotary_emb_a = tuple(item.to(dev_blk_a, non_blocking=True) for item in image_rotary_emb)
                        torch.cuda.synchronize(dev_blk_a)
                        for j, block in enumerate(self.transformer.transformer_blocks[:30]):
                            hidden_states = hidden_states.to(dev_blk_a, non_blocking=True)
                            encoder_hidden_states = encoder_hidden_states.to(dev_blk_a, non_blocking=True)
                            encoder_hidden_states_mask = encoder_hidden_states_mask.to(dev_blk_a, non_blocking=True)
                            torch.cuda.synchronize(dev_blk_a)

                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask,
                                temb=temb_a,
                                image_rotary_emb=image_rotary_emb_a,
                                joint_attention_kwargs=self.attention_kwargs,
                            )

                        temb_b = temb.to(dev_blk_b, non_blocking=True)
                        image_rotary_emb_b = tuple(item.to(dev_blk_b, non_blocking=True) for item in image_rotary_emb)
                        torch.cuda.synchronize(dev_blk_b)
                        for j, block in enumerate(self.transformer.transformer_blocks[30:], start=30):
                            hidden_states = hidden_states.to(dev_blk_b, non_blocking=True)
                            encoder_hidden_states = encoder_hidden_states.to(dev_blk_b, non_blocking=True)
                            encoder_hidden_states_mask = encoder_hidden_states_mask.to(dev_blk_b, non_blocking=True)
                            torch.cuda.synchronize(dev_blk_b)

                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask,
                                temb=temb_b,
                                image_rotary_emb=image_rotary_emb_b,
                                joint_attention_kwargs=self.attention_kwargs,
                            )

                        hidden_states = hidden_states.to(dev_main, non_blocking=True)
                        torch.cuda.synchronize(dev_main)
                        hidden_states = self.transformer.norm_out(hidden_states, temb.to(dev_main))
                        neg_noise_pred = self.transformer.proj_out(hidden_states)

                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # 스케줄러 업데이트
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                torch.cuda.empty_cache()

        self._current_timestep = None

        # 디코드
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)

# ========================
# 1) 입력 데이터 정의
# ========================
def get_scene_data(group_id):
    """특정 group_id에 해당하는 scene 데이터를 반환합니다."""
    scene_data = get_scene_data_by_group_id(group_id)
    if scene_data is None:
        raise ValueError(f"Group ID {group_id}에 해당하는 scene 데이터를 찾을 수 없습니다.")
    return scene_data

def get_goals_data(group_id):
    """특정 group_id에 해당하는 goal_labels 데이터를 반환합니다."""
    goals_data = get_goals_data_by_group_id(group_id)
    if not goals_data:
        raise ValueError(f"Group ID {group_id}에 해당하는 goals 데이터를 찾을 수 없습니다.")
    return goals_data

# ========================
# 2) 프롬프트 빌더
# ========================
def build_prompt(scene: dict, goals: list) -> str:
    floorplan_str = ", ".join(scene.get("floorplan", []))
    objects_str = ", ".join(scene.get("objects", []))
    people_str = ", ".join(scene.get("people", []))

    core = dedent(f"""
    Generate a realistic indoor scene image according to the following description.
    Floorplan includes: {floorplan_str}.
    Objects present in the scene: {objects_str}.
    People present: {people_str}.

    Important conditions:
    - All listed objects must clearly exist in the scene.
    - Each object should appear exactly once (only one instance per object category).
    - The scene should be physically plausible and realistic.
    - People must NOT interact with ANY OBJECTS.
    - People must maintain a safe distance from each other.
    - People must be standing still without any gestures or movements.
    - People MUST keep hands away from all the objects in the scene.
    - Arrange objects on reachable, flat surfaces such as tabletop view.
    - Keep the environment tidy and physically plausible; avoid occluding key objects.
    - Use photorealistic lighting and materials.
    - There is no robot in the scene.
    - There is no text in the scene.

    The scene should consider the feasibility evaluation of these natural language goals:
    """).strip()
    


    labels = {0: "clear instruction", 1: "ambiguous instruction", 2: "infeasible instruction for manipulation task"} # for manipulation robot 없앰.
    goals_block = "\n".join([f'- Goal: "{g["goal"]}" -> Label {g["label"]} ({labels.get(g["label"], "unknown")})' for g in goals])
    return core + "\n" + goals_block

def get_prompts(group_id):
    positive_magic = ", Ultra HD, 4K, cinematic composition, photorealistic, hyperrealistic, highly detailed, natural lighting, realistic textures, volumetric lighting, ambient occlusion, ray tracing, physically based rendering, "
    negative_magic = (
        "hands touching objects, people manipulating objects, object collisions, duplicate objects, multiple instances of same object, "
        "overexposure, blur, motion blur, extra limbs, text, letters, words, writing, distorted text, unrealistic geometry, "
        "missing listed objects, low resolution, cartoon, anime, illustration, painting, drawing, sketch, UI text, watermarks, logos, typography,"
        # "robot, robot parts, mechanical parts, robotic arms, robotic hands, mini robot,"
        # "android, cyborg, machine parts, artificial limbs, metallic body parts, "
        "artificial looking, unnatural colors, unrealistic shadows, poor lighting, "
        "3d rendering artifacts, cgi artifacts, digital artifacts, multiple instances of same category person"
    )
    # negative_magic = (
    #     "text, letters, words, typography, captions, subtitles, labels, signage, logos, watermarks, UI text, "
    #     "books pages with text, menus, screens with text, "
    #     "robot, robot toy, robot parts, android, cyborg, mechanical body parts, robotic arm, robotic hand"
    # )
    
    
    scene = get_scene_data(group_id)
    goals = get_goals_data(group_id)
    prompt = build_prompt(scene, goals) + positive_magic
    negative_prompt = negative_magic
    
    return prompt, negative_prompt

# ========================
# 3) 파이프라인 로딩 및 3-GPU 배치
# ========================
def setup_pipeline():
    model_name = "Qwen/Qwen-Image"

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # 커스텀 파이프라인 사용
    pipe = CustomQwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

    # 디바이스 매핑: cuda:1 메인, 블록 앞 cuda:2, 블록 뒤 cuda:3
    pipe.text_encoder.to("cuda:1")
    pipe.vae.to("cuda:1")

    # 트랜스포머 블록 분할: 절반 기준 30개/나머지
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        if i < 30:
            block.to("cuda:2")
        else:
            block.to("cuda:3")

    # 블록 외 나머지 서브모듈은 메인 디바이스로
    for name, module in pipe.transformer.named_children():
        if name != "transformer_blocks":
            module.to("cuda:1")
    
    return pipe

# ========================
# 4) 해상도/비율 설정 및 생성
# ========================
def generate_image(pipe, prompt, negative_prompt, width, height, seed=42):
    gen = torch.Generator(device="cuda:1").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=8.0, # v2는 4.0 , v3는 7.0, v4는 8.0
        generator=gen,
    ).images[0]
    
    return image, seed

def save_image(image, group_id, width, height, seed):
    import os
    save_dir = "qwen2image_v5"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/scene_group_{group_id}_seed{seed}_{width}x{height}.png"
    image.save(out_path)
    print(f"Saved: {out_path}")
    return out_path

def process_single_group(group_id, pipe, aspect_ratios, seed=42):
    """단일 그룹에 대해 이미지를 생성하고 저장합니다."""
    print(f"\n=== Processing Group ID: {group_id} ===")
    
    try:
        # 프롬프트 생성
        prompt, negative_prompt = get_prompts(group_id)
        
        # 해상도 설정 (기본값: 16:9)
        width, height = aspect_ratios["16:9"]
        
        # 이미지 생성
        image, seed = generate_image(pipe, prompt, negative_prompt, width, height, seed)
        
        # 이미지 저장
        save_image(image, group_id, width, height, seed)
        
        print(f"Successfully processed Group ID: {group_id}")
        return True
        
    except Exception as e:
        print(f"Error processing Group ID {group_id}: {str(e)}")
        return False

def main():
    start_time = time.time()
    
    # 파이프라인 설정
    print("Setting up pipeline...")
    pipe = setup_pipeline()
    
    # 해상도 설정
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "4:3": (1472, 1104),
    }
    
    # 모든 group_id 가져오기
    group_ids = get_all_group_ids()
    print(f"Found {len(group_ids)} groups to process: {group_ids}")
    
    if not group_ids:
        print("처리할 group_id가 없습니다.")
        return
    
    # 각 그룹별로 이미지 생성
    successful_groups = 0
    for i, group_id in tqdm(enumerate(group_ids), total=len(group_ids), desc="Image Generation per Group"):
        
        # 각 그룹마다 다른 시드 사용
        seed = 42 + i
        
        if process_single_group(group_id, pipe, aspect_ratios, seed):
            successful_groups += 1
    
    end_time = time.time()
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful_groups}/{len(group_ids)} groups")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
