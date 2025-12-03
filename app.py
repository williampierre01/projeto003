import os
import io
import re
import base64
import torch
import uvicorn
import spaces
import gradio as gr
from PIL import Image
from contextlib import asynccontextmanager

# Bibliotecas de IA
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Bibliotecas de Servidor e MCP
from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from sse_starlette.sse import EventSourceResponse

# --- 1. CONFIGURAÇÃO DO MODELO COM ACCELERATE ---
# O device_map="auto" usa o 'accelerate' nativamente para gerenciar a VRAM
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

print(f"Carregando {MODEL_ID} na ZeroGPU...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Modelo pronto!")

# --- 2. FUNÇÕES CORE (IA) ---

def clean_latex_output(text):
    """Limpeza robusta do output para garantir LaTeX válido"""
    pattern = r"```(?:latex)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1)
    return text.replace(r"\[", "").replace(r"\]", "").strip()

# Decorator spaces.GPU garante que isso rode na GPU da nuvem
@spaces.GPU
def run_inference(image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Convert the mathematical equation in this image into valid LaTeX code. Output ONLY the LaTeX code. Do not use markdown blocks."}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    # Trim inputs from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return clean_latex_output(output_text)

# --- 3. CONFIGURAÇÃO DO SERVIDOR MCP (LADO DO ROBÔ) ---

mcp_server = Server("qwen-latex-ocr")

@mcp_server.list_tools()
async def list_tools():
    return [
        Tool(
            name="image_to_latex",
            description="Converte uma imagem de equação matemática (base64) para código LaTeX usando Qwen2-VL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "String Base64 da imagem (sem cabeçalho data:image)"
                    }
                },
                "required": ["image_base64"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "image_to_latex":
        try:
            # Decodifica base64 enviado pelo Claude/Agente
            img_data = base64.b64decode(arguments["image_base64"])
            image = Image.open(io.BytesIO(img_data))
            
            # Chama a inferência na GPU
            latex = run_inference(image)
            
            return [TextContent(type="text", text=latex)]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro no processamento: {str(e)}")]
    
    raise ValueError(f"Ferramenta {name} não encontrada")

# --- 4. APLICAÇÃO WEB (FASTAPI + GRADIO) ---

# Gerenciamento de contexto para o servidor MCP
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

# Endpoint SSE para conexão remota do MCP
@app.get("/sse")
async def handle_sse(request: Request):
    async with mcp_server.create_initialization_session() as (read, write):
        async def event_generator():
            async for message in read:
                yield message
        return EventSourceResponse(event_generator())

# Endpoint para receber mensagens do MCP
@app.post("/messages")
async def handle_messages(request: Request):
    # Nota: Em uma implementação real completa, você precisaria conectar 
    # o request de escrita ao stream SSE. 
    # Atualmente, o suporte a SSE remoto nativo no Claude Desktop ainda é experimental,
    # mas esta é a estrutura correta para expor o serviço.
    pass

# --- 5. INTERFACE GRADIO (LADO DO HUMANO) ---
def gradio_interface(image):
    if image is None: return "", ""
    code = run_inference(image)
    return code, f"$${code}$$"

demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
    gr.Markdown("# ⚡ Qwen2-VL + MCP Server")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Upload")
        btn = gr.Button("Extrair LaTeX")
    
    out_code = gr.Code(language="latex", label="Código")
    out_vis = gr.Markdown(label="Visualização")
    
    btn.click(gradio_interface, inputs=img_in, outputs=[out_code, out_vis])

# Monta o Gradio na raiz "/"
app = gr.mount_gradio_app(app, demo, path="/")

# O Hugging Face Spaces procura a variável 'app' no arquivo app.py e roda com uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)