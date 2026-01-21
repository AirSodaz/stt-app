# STT-App Server API 文档

本文档描述了语音转文字（STT）服务器的 REST API 和 WebSocket 接口。

**Base URL**: `http://localhost:8000`

---

## 模型管理

### 获取所有模型

获取所有可用的 ASR 模型及其下载状态。

**请求**
```http
GET /models
```

**响应**
```json
[
  {
    "name": "SenseVoiceSmall",
    "downloaded": true
  },
  {
    "name": "Paraformer",
    "downloaded": false
  }
]
```

---

### 获取已下载模型

获取已下载模型的名称列表。

**请求**
```http
GET /models/downloaded
```

**响应**
```json
["SenseVoiceSmall", "Fun-ASR-Nano"]
```

---

### 下载模型（SSE 流式进度）

启动模型下载并通过 Server-Sent Events (SSE) 流式发送进度。

**请求**
```http
GET /download_model/{model_name}
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `model_name` | path | 要下载的模型名称 |

**响应** (SSE 事件流)
```
data: {"status": "downloading", "progress": 0, "message": "Initializing download..."}

data: {"status": "downloading", "progress": 45, "message": "Downloading model.pt... 45% (402M/893M)"}

data: {"status": "complete", "progress": 100, "message": "Download complete!"}
```

**状态值**
- `downloading`: 下载进行中
- `complete`: 下载完成
- `error`: 下载失败

---

### 下载模型（同步）

同步下载模型（传统接口）。

**请求**
```http
POST /download_model
Content-Type: application/x-www-form-urlencoded

model_name=SenseVoiceSmall
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `model_name` | form | 要下载的模型名称 |

**响应**
```json
{
  "status": "success",
  "message": "Model 'SenseVoiceSmall' downloaded successfully."
}
```

---

## 用户偏好设置

### 获取模型偏好

获取用户保存的模型偏好设置。

**请求**
```http
GET /preference/model
```

**响应**
```json
{
  "model": "SenseVoiceSmall"
}
```

> 如果没有保存偏好或保存的模型未下载，返回 `{"model": null}`

---

### 设置模型偏好

保存用户的模型偏好并加载模型。

**请求**
```http
POST /preference/model
Content-Type: application/x-www-form-urlencoded

model_name=SenseVoiceSmall
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `model_name` | form | 要设置为默认的模型名称 |

**响应**
```json
{
  "status": "success",
  "message": "Model loaded: SenseVoiceSmall"
}
```

---

## 语音转文字

### 转录音频文件

上传音频文件并进行转录。

**请求**
```http
POST /transcribe
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `file` | file | ✓ | - | 要转录的音频文件 |
| `language` | form | ✗ | `"auto"` | 目标语言 |
| `use_itn` | form | ✗ | `false` | 是否使用逆文本规范化 (ITN) |
| `model` | form | ✗ | `null` | 指定使用的模型名称（可选） |

**响应**
```json
{
  "text": "你好，这是转录的文本内容。"
}
```

**错误响应**
```json
{
  "detail": "Speech-to-text model manager not initialized."
}
```

---

### 实时转录 (WebSocket)

WebSocket 端点用于实时流式语音转录。

**连接**
```
ws://localhost:8000/ws/transcribe
```

**协议**

1. 客户端建立 WebSocket 连接
2. 客户端发送音频数据块（二进制 PCM 格式）
3. 服务器返回部分转录结果
4. 发送空数据块表示结束，获取最终结果

**客户端发送**
- 二进制音频数据块（PCM 格式）
- 空字节数组 (`b""`) 表示结束

**服务器响应**

部分结果：
```json
{
  "text": "你好",
  "is_final": false
}
```

最终结果：
```json
{
  "text": "你好，世界！",
  "is_final": true
}
```

错误响应：
```json
{
  "error": "Streaming inference error message"
}
```

> **注意**: WebSocket 流式转录默认使用 Paraformer 模型。

---

## 错误处理

所有 API 在发生错误时返回标准的 HTTP 错误响应：

| 状态码 | 描述 |
|--------|------|
| `400` | 请求参数错误（如未知模型名称） |
| `500` | 服务器内部错误 |

**错误响应格式**
```json
{
  "detail": "错误信息描述"
}
```

---

## 支持的模型

| 模型名称 | 描述 | 流式支持 |
|----------|------|----------|
| `SenseVoiceSmall` | SenseVoice 小型模型 | ✗ |
| `Fun-ASR-Nano` | FunASR Nano 模型 | ✗ |
| `Paraformer` | Paraformer 中文流式模型 | ✓ |

---

## 使用示例

### cURL 示例

**转录音频文件**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "language=zh" \
  -F "use_itn=false"
```

**获取模型列表**
```bash
curl "http://localhost:8000/models"
```

**下载模型（带进度）**
```bash
curl -N "http://localhost:8000/download_model/SenseVoiceSmall"
```

### JavaScript 示例

**转录音频**
```javascript
const formData = new FormData();
formData.append('file', audioBlob, 'audio.wav');
formData.append('language', 'auto');

const response = await fetch('http://localhost:8000/transcribe', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.text);
```

**实时转录 (WebSocket)**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/transcribe');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.text) {
    console.log(data.is_final ? '最终:' : '部分:', data.text);
  }
};

// 发送音频数据块
ws.send(audioChunk); // ArrayBuffer

// 结束流式转录
ws.send(new ArrayBuffer(0));
```
