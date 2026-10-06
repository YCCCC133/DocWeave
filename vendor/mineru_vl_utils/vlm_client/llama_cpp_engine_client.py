import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Sequence

from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from mineru_llama_cpp import GenerateResult as LlamaCppGenerateResult

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ImageType,
    RequestError,
    SamplingParams,
    ServerError,
    VlmClient,
)
from .utils import (
    VLM_PREDICT_DESC,
    aio_image_to_bytes_list_and_format,
    gather_tasks,
    get_image_data_url,
    image_to_bytes_list_and_format,
)


class LlamaCppEngineVlmClient(VlmClient):
    def __init__(
        self,
        llama_cpp_engine,  # mineru_llama_cpp.Engine instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 100,
        debug: bool = False,
        use_tqdm: bool = True,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        try:
            from mineru_llama_cpp import EngineError as LlamaCppEngineError
            from mineru_llama_cpp import Engine as LlamaCppEngine
            from mineru_llama_cpp import InvalidRequestError as LlamaCppInvalidRequestError
            from mineru_llama_cpp import SamplingParams as LlamaCppSamplingParams
        except ImportError:
            raise ImportError("Please install mineru-llama-cpp to use LlamaCppEngineVlmClient.")

        if not llama_cpp_engine:
            raise ValueError("llama_cpp_engine is None.")
        if not isinstance(llama_cpp_engine, LlamaCppEngine):
            raise ValueError(f"llama_cpp_engine must be an instance of {LlamaCppEngine}.")

        self.llama_cpp_engine = llama_cpp_engine
        self.LlamaCppSamplingParams = LlamaCppSamplingParams
        self.LlamaCppInvalidRequestError = LlamaCppInvalidRequestError
        self.LlamaCppEngineError = LlamaCppEngineError
        self.max_concurrency = max_concurrency
        self.debug = debug
        self.use_tqdm = use_tqdm

    def build_messages(self, image_urls: list[str], prompt: str) -> list[dict]:
        prompt = prompt or self.prompt
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
            prompt_parts = prompt.split("<image>", len(image_urls))
            user_messages = []
            for i in range(max(len(prompt_parts), len(image_urls))):
                if i < len(prompt_parts) and prompt_parts[i]:
                    user_messages.append({"type": "text", "text": prompt_parts[i]})
                if i < len(image_urls):
                    user_messages.append({"type": "image_url", "image_url": {"url": image_urls[i]}})
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                *({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls),
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                *({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls),
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    @staticmethod
    def _default_generation_budget(image: ImageType, prompt: str) -> int:
        """Bound runaway decoding without treating Chinese text as a formula task.

        Image area is only a conservative proxy for text density. Large or
        unknown-size text crops retain a larger budget; hitting any budget is
        an error, never a successfully recognized fragment.
        """
        task = prompt.strip().lower()
        if task.startswith("layout detection:"):
            return 4096
        images = image if isinstance(image, (list, tuple)) else [image]
        areas = []
        for item in images:
            size = getattr(item, "size", None)
            if not (isinstance(size, tuple) and len(size) == 2 and
                    all(isinstance(edge, int) and edge > 0 for edge in size)):
                areas = []
                break
            areas.append(size[0] * size[1])
        area = sum(areas) if areas else None
        if task.startswith("formula recognition:"):
            return 2048 if area is not None and area <= 600_000 else 4096
        if task.startswith("table recognition:"):
            return 8192
        return 4096 if area is not None and area <= 1_500_000 else 8192

    def build_llama_cpp_sampling_params(
        self, sampling_params: SamplingParams | None, *, image: ImageType = None, prompt: str = ""
    ):
        sp = self.build_sampling_params(sampling_params)

        # NOTE: no_repeat_ngram_size has no equivalent in mineru_llama_cpp.SamplingParams
        # (its own docstring deliberately excludes n-gram-based repetition
        # suppression samplers) -- silently dropped here, same as
        # MlxVlmClient/LmdeployEngineVlmClient, which never read this field either.
        #
        # Engine does not expose n_ctx. Use a finite task budget only when the
        # caller supplied no max_new_tokens, rather than decoding to 32k on
        # abnormal output. Preserve explicit request/client-level settings.
        max_new_tokens = sp.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self._default_generation_budget(image, prompt or self.prompt)
        llama_cpp_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repeat_penalty": sp.repetition_penalty,
            "n_predict": max_new_tokens,
        }
        return self.LlamaCppSamplingParams(**{k: v for k, v in llama_cpp_sp_dict.items() if v is not None})

    def get_output_content(self, result: "LlamaCppGenerateResult", *, require_complete: bool = False) -> str:
        if result.finish_reason == "length":
            if require_complete or not self.allow_truncated_content:
                raise RequestError("The output was truncated due to length limit.")
            else:
                logger.warning("The output was truncated due to length limit.")
        elif result.finish_reason != "stop":
            raise RequestError(f"Unexpected finish reason: {result.finish_reason}")
        return result.content

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        image_bytes, image_format = image_to_bytes_list_and_format(image)
        image_urls = [get_image_data_url(im, image_format) for im in image_bytes]
        messages = self.build_messages(image_urls, prompt)
        default_budget = self.build_sampling_params(sampling_params).max_new_tokens is None
        llama_cpp_sp = self.build_llama_cpp_sampling_params(sampling_params, image=image, prompt=prompt)

        if self.debug:
            logger.debug("Messages: {}", messages)

        try:
            result = self.llama_cpp_engine.generate(messages, llama_cpp_sp)
        except self.LlamaCppInvalidRequestError as e:
            raise RequestError(str(e))
        except self.LlamaCppEngineError as e:
            raise ServerError(str(e))

        if self.debug:
            logger.debug("Result: {}", result)

        return self.get_output_content(result, require_complete=default_budget)

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        """并发提取内容，按完成请求更新进度，并保持输入对应的结果顺序。"""
        images_len = len(images)
        if isinstance(prompts, str):
            prompts = [prompts] * images_len
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * images_len
        if not isinstance(priority, Sequence):
            priority = [priority] * images_len

        assert len(prompts) == images_len, "Length of prompts and images must match."
        assert len(sampling_params) == images_len, "Length of sampling_params and images must match."
        assert len(priority) == images_len, "Length of priority and images must match."
        if not images_len:
            return []

        # Engine.generate() releases the GIL in its C++ layer (see
        # mineru-llama-cpp's test_concurrency.py), so a plain thread pool lets
        # concurrent calls actually decode in parallel across the engine's
        # n_parallel slots, instead of serializing on the GIL.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(self.predict, *args): index
                for index, args in enumerate(zip(images, prompts, sampling_params, priority))
            }
            results = [""] * images_len
            with tqdm(total=images_len, desc=VLM_PREDICT_DESC, disable=not self.use_tqdm) as pbar:
                for future in as_completed(futures):
                    # 完成顺序只影响进度显示，内容仍回填到原始请求对应的位置。
                    results[futures[future]] = future.result()
                    pbar.update(1)
        return results

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        image_bytes, image_format = await aio_image_to_bytes_list_and_format(image)
        image_urls = [get_image_data_url(im, image_format) for im in image_bytes]
        messages = self.build_messages(image_urls, prompt)
        default_budget = self.build_sampling_params(sampling_params).max_new_tokens is None
        llama_cpp_sp = self.build_llama_cpp_sampling_params(sampling_params, image=image, prompt=prompt)

        if self.debug:
            logger.debug("Messages: {}", messages)

        try:
            result = await self.llama_cpp_engine.agenerate(messages, llama_cpp_sp)
        except self.LlamaCppInvalidRequestError as e:
            raise RequestError(str(e))
        except self.LlamaCppEngineError as e:
            raise ServerError(str(e))

        if self.debug:
            logger.debug("Result: {}", result)

        return self.get_output_content(result, require_complete=default_budget)

    async def aio_batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            image: ImageType,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                )

        return await gather_tasks(
            tasks=[
                predict_with_semaphore(*args)
                for args in zip(images, prompts, sampling_params, priority)
            ],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )
