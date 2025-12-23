# cosyvoice_service.py - 批处理优化版本
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice, CosyVoice2, CosyVoice3
from typing import Optional
import os
import logging
import tempfile
import soundfile as sf
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import List
import uuid

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*weight_norm.*')

app = FastAPI()

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

model_dir = "/home/ec2-user/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
USE_FP16 = os.getenv('COSYVOICE_FP16', 'true').lower() == 'true'

logger.warning(f"Loading model from: {model_dir}")
cosyvoice = AutoModel(model_dir=model_dir, fp16=USE_FP16)
model_type = type(cosyvoice).__name__

default_prompt_wav = "/home/ec2-user/CosyVoice/asset/paimon_prompt.wav"
default_prompt_text = "等等,算起来…今天是不是就是连续打工的第三天了?现在正是午饭时间!"

# 批处理队列
@dataclass
class TTSTask:
    task_id: str
    text: str
    instruction: Optional[str]
    prompt_wav_path: str
    prompt_text: Optional[str]
    result_queue: asyncio.Queue
    
class TTSBatchProcessor:
    def __init__(self, model, batch_size=4, max_wait_ms=50):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.task_queue = asyncio.Queue()
        self.processing = False
        
    async def start(self):
        """启动批处理循环"""
        if self.processing:
            return
        self.processing = True
        asyncio.create_task(self._process_loop())
        
    async def _process_loop(self):
        """批处理主循环"""
        while self.processing:
            tasks = []
            deadline = asyncio.get_event_loop().time() + (self.max_wait_ms / 1000)
            
            # 收集一批任务
            while len(tasks) < self.batch_size:
                timeout = max(0, deadline - asyncio.get_event_loop().time())
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=timeout)
                    tasks.append(task)
                except asyncio.TimeoutError:
                    break
                    
            if not tasks:
                await asyncio.sleep(0.01)
                continue
                
            # 批量处理
            await self._process_batch(tasks)
            
    async def _process_batch(self, tasks: List[TTSTask]):
        """处理一批任务"""
        logger.debug(f"Processing batch of {len(tasks)} tasks")
        
        # 对于CosyVoice模型,目前还是需要逐个处理
        # 但可以优化为异步执行,减少阻塞
        for task in tasks:
            try:
                # 在executor中运行推理
                loop = asyncio.get_event_loop()
                audio_chunks = await loop.run_in_executor(
                    None, 
                    self._synthesize_sync, 
                    task
                )
                await task.result_queue.put({"status": "success", "audio": audio_chunks})
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                await task.result_queue.put({"status": "error", "error": str(e)})
                
    def _synthesize_sync(self, task: TTSTask) -> List[np.ndarray]:
        """同步推理方法"""
        chunks = []
        
        if isinstance(self.model, (CosyVoice2, CosyVoice3)):
            if task.instruction:
                inference_gen = self.model.inference_instruct2(
                    task.text,
                    task.instruction,
                    task.prompt_wav_path,
                    stream=True
                )
            else:
                prompt = 'You are a helpful assistant.<|endofprompt|>' + (task.prompt_text or default_prompt_text)
                inference_gen = self.model.inference_zero_shot(
                    task.text,
                    prompt,
                    task.prompt_wav_path,
                    stream=True
                )
                
            for result in inference_gen:
                chunks.append(result["tts_speech"].squeeze().cpu().numpy())
                
        return chunks
        
    async def submit_task(self, task: TTSTask):
        """提交任务到队列"""
        await self.task_queue.put(task)

# 全局批处理器
batch_processor = TTSBatchProcessor(cosyvoice, batch_size=4, max_wait_ms=50)

@app.on_event("startup")
async def startup_event():
    """启动时初始化批处理器"""
    await batch_processor.start()

@app.post("/synthesize")
async def synthesize_streaming(
    text: str = Form(...),
    instruction: Optional[str] = Form(None),
    prompt_text: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None)
):
    """批处理优化的TTS接口"""
    try:
        # 处理音频文件
        temp_wav_path = None
        if prompt_wav:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                content = await prompt_wav.read()
                temp_file.write(content)
                temp_wav_path = temp_file.name
        elif os.path.exists(default_prompt_wav):
            temp_wav_path = default_prompt_wav
        else:
            raise HTTPException(status_code=400, detail="No voice reference")
            
        # 创建任务
        task_id = str(uuid.uuid4())
        result_queue = asyncio.Queue()
        task = TTSTask(
            task_id=task_id,
            text=text,
            instruction=instruction,
            prompt_wav_path=os.path.abspath(temp_wav_path),
            prompt_text=prompt_text,
            result_queue=result_queue
        )
        
        # 提交到批处理队列
        await batch_processor.submit_task(task)
        logger.debug(f"Task {task_id} submitted")
        
        # 等待结果
        result = await result_queue.get()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        # 流式返回音频
        async def audio_stream():
            try:
                for chunk in result["audio"]:
                    yield chunk.tobytes()
            finally:
                if prompt_wav and temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.unlink(temp_wav_path)
                    except:
                        pass
                        
        return StreamingResponse(audio_stream(), media_type="audio/wav")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_type": model_type}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50000,
        workers=1,              # 批处理模式使用单worker
        limit_concurrency=100,  # 增加并发
        timeout_keep_alive=30,
        backlog=2048,
        log_level="warning"
    )
