import sys
import io

# 1. 🌟 极致的全局编码重置：确保 stdout / stderr 哪怕在最顽固的 ASCII 环境下也强行使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import jwt
import datetime
import sqlite3
import os
import base64
import numpy as np
import cv2
import tempfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 🚀 引入 RAG 核心库
import chromadb
from openai import OpenAI

# 🤖 引入 YOLO 和表情识别
from ultralytics import YOLO
# DeepFace is imported lazily inside analyze_emotion() due to segfault on Python 3.13

app = FastAPI()

# 密钥（Secret Key）：用于加密和解密 Token
SECRET_KEY = "lux_secret_key_123456" 
# 加密算法
ALGORITHM = "HS256"
import json

def verify_token(request: Request):
    """Extract and verify JWT from Authorization header. Returns user_id or raises 401."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少认证令牌")
    token = auth.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="令牌已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的认证令牌")

def ok(data=None, message="操作成功"):
    return {"status": 200, "message": message, "data": data}

# 2. 🌟 挂载静态文件目录（头像等）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. 🌟 解决 CORS 跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有前端源
    allow_credentials=True,
    allow_methods=["*"],          # 允许所有请求方法
    allow_headers=["*"],          # 允许所有请求头
)

# 🌐 首页 — 四宫格导航
@app.get("/home")
async def home():
    return FileResponse("app_home.html")

# 🌐 根路由 — 返回浏览器版聊天应用
@app.get("/")
async def root():
    return FileResponse("app.html")

# 🌐 实时表情识别版
@app.get("/realtime")
async def realtime():
    return FileResponse("app_realtime.html")

# 🌐 绘画心理分析版
@app.get("/draw")
async def draw():
    return FileResponse("app_draw.html")

# 🌐 CBT-I 失眠认知行为治疗
@app.get("/cbti")
async def cbti():
    return FileResponse("app_cbti.html")

# 3. 🌟 建立 MySQL 数据库连接的方法
def get_db_connection():
    conn = sqlite3.connect("psychology_assistant.db")
    conn.row_factory = sqlite3.Row  # 让查询结果返回字典格式
    return conn

# 4. 🌟 初始化 ChromaDB 和 大模型客户端 (DeepSeek)
try:
    # 连接本地 Docker 里的 ChromaDB 向量数据库
    chroma_client = chromadb.PersistentClient(path='./chroma_db')
    # 获取之前初始化好并灌入了专业心理文章的数据集
    collection = chroma_client.get_collection(name="psychology_rules")
    print("[ChromaDB] Connected successfully and got psychology_rules collection.")
except Exception as e:
    # 采用安全编码打印，防止异常信息中含中文字符导致 print 崩溃
    print(f"[ChromaDB] Warning: Connection failed. Error info safely converted: {repr(e)}")
    collection = None

# 初始化 OpenAI 客户端（这里以 DeepSeek 为例）
# 💡 提示：将系统环境变量作为首选，防止 Key 泄漏或硬编码失效
ai_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key"),
    base_url="https://api.deepseek.com/v1"  # 标准 DeepSeek 官方接口地址
)

# 🤖 初始化 YOLOv8 模型（用于人脸检测）
# 首次运行会自动下载 yolov8n.pt 到本地
try:
    yolo_model = YOLO("yolov8n.pt")
    print("[YOLO] YOLOv8n model loaded successfully for face detection.")
except Exception as e:
    print(f"[YOLO] Warning: Failed to load YOLO model: {repr(e)}")
    yolo_model = None

# 情绪标签中英文映射
EMOTION_LABELS_ZH = {
    "angry": "愤怒 😠",
    "disgust": "厌恶 🤢",
    "fear": "恐惧 😨",
    "happy": "开心 😊",
    "sad": "悲伤 😢",
    "surprise": "惊讶 😲",
    "neutral": "平静 😐"
}

# DeepFace 情绪分析结果 → 心理咨询线索映射
EMOTION_ADVICE_HINTS = {
    "angry": "用户当前情绪偏愤怒，建议先引导深呼吸放松，用温和的语气询问愤怒的源头。",
    "disgust": "用户可能对某些事物感到厌恶，建议避免直接追问，先建立信任。",
    "fear": "用户当前有恐惧/焦虑情绪，建议使用CBT认知重构技巧，帮助用户区分现实威胁和想象威胁。",
    "happy": "用户情绪状态良好，是建立积极心理资源的好时机，可以引导其分享快乐的事。",
    "sad": "用户可能处于悲伤或低落状态，建议使用共情倾听技巧，肯定其感受的合理性。",
    "surprise": "用户处于惊讶状态，可以顺势引导其表达当下的感受。",
    "neutral": "用户情绪平稳，适合进行常规心理咨询对话。"
}


# 5. 🌟 接收前端的 POST 请求（完美兼容：账号密码登录、手机验证码登录、更新资料）
@app.post("/login")
def login_or_update(payload: dict):
    # 安全打印前端发送的原始数据
    try:
        print("Received payload:", payload)
    except Exception:
        print("Received payload containing unprintable characters.")
    
    # 提取前端传过来的字段
    user_id = payload.get("id")
    username = payload.get("username")
    password = payload.get("password")
    
    # 手机验证码登录相关字段
    tel = payload.get("tel")
    verify_code = payload.get("verify_code")
    
    # 修改资料相关字段
    nickname = payload.get("nickname")
    gender = payload.get("gender")
    bio = payload.get("bio")          # 前端的 bio 对应数据库中的 sign (个性签名)
    avatar = payload.get("avatar")

    # 建立数据库连接
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # ---- 情况 A：用户使用【账号密码登录】（传了 username 且没有有效 user_id） ----
        if username and (user_id is None or int(user_id) == 0):
            sql = "SELECT * FROM user_account WHERE username = ? AND password = ?"
            cursor.execute(sql, (username, password))
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(status_code=400, detail="账号或密码错误")
            
            user_id = user["id"]
            nickname = user["nickname"]
            tel = user["phone"]
            gender = user["gender"]
            bio = user["sign"]
            avatar = user["avatar"]

        # ---- 情况 B：用户使用【手机验证码登录】（传了 tel 和 verify_code，且没有有效 user_id） ----
        elif tel and verify_code and (user_id is None or int(user_id) == 0):
            # 先去数据库查这个手机号是否已经存在
            sql = "SELECT * FROM user_account WHERE phone = ?"
            cursor.execute(sql, (tel,))
            user = cursor.fetchone()
            
            if user:
                # 如果手机号存在，直接获取用户信息
                user_id = user["id"]
                nickname = user["nickname"]
                gender = user["gender"]
                bio = user["sign"]
                avatar = user["avatar"]
            else:
                # 如果手机号不存在，自动在数据库中“注册”一个新用户
                default_nickname = f"用户_{tel[-4:]}"
                default_avatar = "/static/7.jpeg"
                insert_sql = """
                    INSERT INTO user_account (phone, nickname, avatar, gender, sign) 
                    VALUES (?, ?, ?, ?, ?)
                """
                cursor.execute(insert_sql, (tel, default_nickname, default_avatar, "女", "新注册的心理助手用户"))
                conn.commit()  # 提交到数据库保存
                
                # 获取刚刚插入的新用户的自增 ID
                user_id = cursor.lastrowid
                nickname = default_nickname
                avatar = default_avatar
                gender = "女"
                bio = "新注册的心理助手用户"

        # ---- 情况 C：用户正在【修改个人资料】（已经登录，传了大于 0 的真实用户 ID） ----
        elif user_id and int(user_id) > 0:
            # 执行 UPDATE 语句更新数据库
            sql = """
                UPDATE user_account 
                SET nickname = ?, phone = ?, gender = ?, sign = ?, avatar = ? 
                WHERE id = ?
            """
            cursor.execute(sql, (nickname, tel, gender, bio, avatar, user_id))
            conn.commit()
            
            # 重新查询一次最新数据返回给前端，确保数据一致
            cursor.execute("SELECT * FROM user_account WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            if user:
                nickname = user["nickname"]
                tel = user["phone"]
                gender = user["gender"]
                bio = user["sign"]
                avatar = user["avatar"]

        # ---- 情况 D：异常数据（例如没登录却传了 id=0 修改资料） ----
        else:
            raise HTTPException(
                status_code=400, 
                detail="请求参数不完整或未登录！请先在登录界面输入账号密码登录后再修改资料。"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        conn.rollback()
        print("MySQL Error safely caught:", repr(e))
        raise HTTPException(status_code=500, detail="数据库内部错误")
    finally:
        cursor.close()
        conn.close()

    # 🌟 准备 JWT 载荷（Payload）
    jwt_payload = {
        "user_id": user_id,
        "nickname": nickname,
        "tel": tel,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1) 
    }
    
    # 🌟 动态生成加密 Token
    real_token = jwt.encode(jwt_payload, SECRET_KEY, algorithm=ALGORITHM)
    
    return {
        "status": 200,
        "message": "操作成功",
        "data100": {
            "token": real_token,
            "user": {
                "id": user_id,
                "nickname": nickname,
                "tel": tel,
                "gender": gender,
                "bio": bio,
                "avatar": avatar
            }
        }
    }


# 6. 🤖 YOLO + DeepFace 表情识别 POST 路由
@app.post("/analyze_emotion")
async def analyze_emotion(payload: dict):
    """
    接收前端传来的 base64 图片，用 YOLO 检测人脸，DeepFace 分析情绪。
    返回情绪分析结果，供心理咨询对话参考。
    """
    image_b64 = payload.get("image", "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="图片数据 (image, base64) 不能为空")

    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO 模型未成功加载，无法进行人脸检测")

    try:
        # ---- 步骤 A：解码 base64 → OpenCV 图像 ----
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片数据，请确认传入了有效的 base64 图片")

        # ---- 步骤 B：保存为临时文件 ----
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, img)

        faces_found = []
        emotion_summary = None

        try:
            # ---- 步骤 C：用 DeepFace 自带的 RetinaFace 检测人脸 + 分析情绪 ----
            # DeepFace 内置多个检测器: retinaface, mtcnn, opencv, ssd, dlib
            # retinaface 最准，mtcnn 次之
            from deepface import DeepFace
            emotion_result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['emotion'],
                detector_backend='retinaface',  # 最准的人脸检测器
                enforce_detection=True,
                silent=True
            )

            # DeepFace 返回 list，每张脸一个结果
            if isinstance(emotion_result, list):
                for i, face_data in enumerate(emotion_result):
                    region = face_data.get('region', {})
                    dominant = face_data.get('dominant_emotion', 'neutral')
                    scores = face_data.get('emotion', {})
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

                    faces_found.append({
                        "face_index": i,
                        "dominant_emotion": dominant,
                        "dominant_emotion_zh": EMOTION_LABELS_ZH.get(dominant, dominant),
                        "emotion_scores": {k: round(float(v), 3) for k, v in sorted_scores},
                        "advice_hint": EMOTION_ADVICE_HINTS.get(dominant, ""),
                        "face_region": {
                            "x": region.get('x', 0),
                            "y": region.get('y', 0),
                            "w": region.get('w', 0),
                            "h": region.get('h', 0)
                        }
                    })
            else:
                # 单张脸
                region = emotion_result.get('region', {})
                dominant = emotion_result.get('dominant_emotion', 'neutral')
                scores = emotion_result.get('emotion', {})
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                faces_found.append({
                    "face_index": 0,
                    "dominant_emotion": dominant,
                    "dominant_emotion_zh": EMOTION_LABELS_ZH.get(dominant, dominant),
                    "emotion_scores": {k: round(float(v), 3) for k, v in sorted_scores},
                    "advice_hint": EMOTION_ADVICE_HINTS.get(dominant, ""),
                    "face_region": {
                        "x": region.get('x', 0),
                        "y": region.get('y', 0),
                        "w": region.get('w', 0),
                        "h": region.get('h', 0)
                    }
                })

        except Exception as deepface_error:
            # RetinaFace 可能太严格了，用 opencv 试一次
            print(f"[Emotion] RetinaFace failed: {repr(deepface_error)}, trying opencv backend...")
            try:
                emotion_result = DeepFace.analyze(
                    img_path=tmp_path,
                    actions=['emotion'],
                    detector_backend='opencv',
                    enforce_detection=True,
                    silent=True
                )
                if isinstance(emotion_result, list):
                    for i, face_data in enumerate(emotion_result):
                        region = face_data.get('region', {})
                        dominant = face_data.get('dominant_emotion', 'neutral')
                        scores = face_data.get('emotion', {})
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        faces_found.append({
                            "face_index": i,
                            "dominant_emotion": dominant,
                            "dominant_emotion_zh": EMOTION_LABELS_ZH.get(dominant, dominant),
                            "emotion_scores": {k: round(float(v), 3) for k, v in sorted_scores},
                            "advice_hint": EMOTION_ADVICE_HINTS.get(dominant, ""),
                            "detection_method": "opencv (fallback)"
                        })
                else:
                    region = emotion_result.get('region', {})
                    dominant = emotion_result.get('dominant_emotion', 'neutral')
                    scores = emotion_result.get('emotion', {})
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    faces_found.append({
                        "face_index": 0,
                        "dominant_emotion": dominant,
                        "dominant_emotion_zh": EMOTION_LABELS_ZH.get(dominant, dominant),
                        "emotion_scores": {k: round(float(v), 3) for k, v in sorted_scores},
                        "advice_hint": EMOTION_ADVICE_HINTS.get(dominant, ""),
                        "detection_method": "opencv (fallback)"
                    })
            except Exception as fallback_error:
                print(f"[Emotion] All detectors failed. RetinaFace: {repr(deepface_error)}, OpenCV: {repr(fallback_error)}")

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # 构建 summary
        if faces_found:
            emotion_summary = {
                "primary_emotion": faces_found[0]["dominant_emotion"],
                "primary_emotion_zh": faces_found[0]["dominant_emotion_zh"],
                "advice_hint": faces_found[0]["advice_hint"]
            }

        if not faces_found:
            return {
                "status": 200,
                "message": "未检测到清晰人脸",
                "data": {
                    "faces_detected": 0,
                    "faces": [],
                    "emotion_summary": None,
                    "hint": "请确保光线充足、正脸面对镜头，再试一次"
                }
            }

        return {
            "status": 200,
            "message": "情绪分析完成",
            "data": {
                "faces_detected": len(faces_found),
                "faces": faces_found,
                "emotion_summary": emotion_summary
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"情绪分析失败: {str(e)}\n堆栈: {error_details}")


# 7. 🌟 RAG 智能对话 POST 路由（可选融合表情分析）
@app.post("/chat")
def r_a_g_chat(payload: dict):
    # 接收前端发送的用户聊天消息
    user_message = payload.get("message", "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    if collection is None:
        raise HTTPException(status_code=500, detail="向量数据库连接未成功初始化，请检查后台日志")

    # 🌟 接收可选的表情分析上下文
    emotion_context = payload.get("emotion_context", None)

    try:
        # ---- 【步骤 A：去向量数据库里检索心理学相关知识】 ----
        results = collection.query(
            query_texts=[user_message],
            n_results=1
        )

        # 提取查出来的文章内容
        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_doc = retrieved_docs[0] if retrieved_docs else "无相关参考资料"

        # 💡 安全日志输出：将有中文字符的信息限制在 ASCII 打印保护下，不使用会引发崩溃的 Emoji 和原生中文字符拼接
        print(f"[RAG Query] Received message length: {len(user_message)}")
        print(f"[RAG Doc Match] Successfully retrieved context doc, length: {len(retrieved_doc)}")

        # ---- 【步骤 B：拼接 Prompt（专业知识 + 情绪上下文）】 ----
        system_prompt = (
            "你是一位温柔、专业、极具共情力的心理咨询师。\n"
            "在回答用户问题时，请务必严格参考我们提供的【本地参考资料】，"
            "结合资料中的专业心理学步骤和理论，用极其温和、体贴、平缓的语气，"
            "一步一步引导并疏导用户的情绪。不要使用生硬的说教，像朋友一样给予其温暖和可操作的建议。\n\n"
            f"【本地参考资料（仅供参考和融合）】:\n{retrieved_doc}"
        )

        # 🌟 如果有情绪分析结果，加入提示
        if emotion_context:
            primary = emotion_context.get("primary_emotion", "")
            hint = emotion_context.get("advice_hint", "")
            if primary:
                system_prompt += (
                    f"\n\n【重要：用户当前情绪状态分析】:\n"
                    f"通过面部表情识别，用户当前主要情绪为: {EMOTION_LABELS_ZH.get(primary, primary)}。\n"
                    f"咨询建议: {hint}\n"
                    f"请在回复中自然地融入对该情绪的共情和处理技巧，但不要直接说'我检测到你的表情'等暴露技术细节的话。"
                )

        # ---- 【步骤 C：调用 DeepSeek 接口，让 AI 看着参考资料来作答】 ----
        response = ai_client.chat.completions.create(
            model="deepseek-chat",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        ai_reply = response.choices[0].message.content

        # ---- 【步骤 D：把 AI 暖心的回答和检索的数据来源，一并返回给前端】 ----
        return {
            "status": 200,
            "message": "发送成功",
            "data": {
                "reply": ai_reply,
                "source": retrieved_doc  # 把检索源也返给前端
            }
        }

    except Exception as e:
        # 💡 临时 Debug：直接把真实报错丢给 Thunder Client 暴露出来！
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"真实报错详情: {str(e)}\n堆栈: {error_details}")


# 8. 🎨 绘画心理分析 POST 路由
@app.post("/analyze_drawing")
async def analyze_drawing(payload: dict):
    """
    接收绘画图片(base64) + 提取的画面特征，
    由 AI 进行专业的绘画心理分析（艺术治疗方向）。
    """
    image_b64 = payload.get("image", "")
    features = payload.get("features", {})

    if not image_b64:
        raise HTTPException(status_code=400, detail="绘画图片数据不能为空")

    try:
        # ---- 构建详细的绘画特征描述（用于 AI 分析） ----
        cov = features.get('coverage_pct', 0)
        size = features.get('size_hint', 'medium')
        h_pos = features.get('horizontal_position', 'center')
        v_pos = features.get('vertical_position', 'center')
        balance_h = features.get('balance_h', 'balanced')
        balance_v = features.get('balance_v', 'balanced')
        is_portrait = features.get('is_portrait', False)
        is_landscape = features.get('is_landscape', False)
        stroke = features.get('stroke_style', '未知')
        dom_color = features.get('dominant_color', '未知')
        color_count = features.get('color_count', 0)
        palette = features.get('color_palette', [])
        warm = features.get('warm_ratio', 0)
        cool = features.get('cool_ratio', 0)
        dark = features.get('dark_ratio', 0)
        color_tone = features.get('color_tone', 'neutral')
        focus_zone = features.get('focus_zone', '未知')
        empty_zone = features.get('empty_zone', 'none')
        top_half = features.get('top_half_pct', 50)
        bottom_half = features.get('bottom_half_pct', 50)
        left_half = features.get('left_half_pct', 50)
        right_half = features.get('right_half_pct', 50)
        center_zone = features.get('center_zone_pct', 0)
        third_h = features.get('third_h', [33, 33, 33])
        third_v = features.get('third_v', [33, 33, 33])

        palette_str = "、".join([f"{c['name']}({c['pct']}%)" for c in palette[:5]]) if palette else "无"

        feature_desc = f"""
=== 绘画量化特征报告 ===

【尺寸与占比】
- 绘画占画面 {cov}%，属于{size}级别
- {"画面非常小，笔触极为克制" if size == 'tiny' else "画面偏小，有留白空间" if size == 'small' else "画面大小适中" if size == 'medium' else "画面较大，有表达欲望" if size == 'large' else "画面充满画布，表达强烈"}

【空间分布 - 艺术治疗空间象征学】
- 重心水平: {h_pos}（{"左=过去/内向/母性；右=未来/外向/父性" if h_pos == 'left' else "中心=平衡/当下" if h_pos == 'center' else "右=进取/外向/未来导向" if h_pos == 'right' else ''}）
- 重心垂直: {v_pos}（{"上=精神/幻想/乐观；下=现实/物质/根基" if v_pos == 'top' else "中=现实感/稳定" if v_pos == 'center' else "下=务实/不安全感/求稳" if v_pos == 'bottom' else ''}）
- 左右平衡: {balance_h}（{'画面重心偏左，可能更关注内心世界和过往经历' if balance_h == 'left_heavy' else '画面重心偏右，可能更关注外部世界和未来发展' if balance_h == 'right_heavy' else '左右均衡' if balance_h == 'balanced' else ''}）
- 上下平衡: {balance_v}（{'画面上重下轻，可能偏理想主义或精神追求' if balance_v == 'top_heavy' else '画面下重上轻，可能偏务实或有一定安全感需求' if balance_v == 'bottom_heavy' else '上下均衡' if balance_v == 'balanced' else ''}）
- 上半部分占{top_half}%，下半部分占{bottom_half}%
- 左半部分占{left_half}%，右半部分占{right_half}%
- 中心区域占{center_zone}%（中心的高占比通常与自我关注度和核心自我感相关）
- 最集中区域: {focus_zone}
- 空白区域: {empty_zone}

【构图形态】
- 方向: {'纵向（暗示内向、情绪化、精神性）' if is_portrait else '横向（暗示外向、理性、社交性）' if is_landscape else '接近方形（暗示稳定、秩序感）'}
- 笔触风格: {stroke}（线条型=克制理性；色块型=感性直觉；结合型=灵活平衡）

【色彩心理分析】
- 主色调: {dom_color}
- 色系丰富度: {color_count}种（{'单色=克制专注' if color_count <= 1 else '适量' if color_count <= 3 else '丰富=情感世界活跃' if color_count <= 6 else '非常丰富=情感强烈或创造力旺盛'}）
- 调色板: {palette_str}
- 暖色比例: {warm}%（暖色关联情感表达、热情、冲动）
- 冷色比例: {cool}%（冷色关联理性、克制、内省）
- 暗色比例: {dark}%（暗色可能与沉重情绪、深度思考或压抑有关）
- 整体色调: {'偏暖，可能情感表达较强' if color_tone == 'warm' else '偏冷，可能偏理性克制' if color_tone == 'cool' else '中性温和'}
"""

        # ---- 构建专业心理分析提示词 ----
        system_prompt = (
            "你是一位拥有25年临床经验的艺术治疗心理分析师。你精通：\n"
            "- 房树人投射测验(HTP)的深度解读\n"
            "- 荣格分析心理学中的曼陀罗和自由绘画分析\n"
            "- 色彩心理学（吕舍尔色彩测试框架）\n"
            "- 空间象征学（格伦沃尔德空间图示理论）\n"
            "- 格式塔心理学在绘画分析中的应用\n\n"
            "现在，你收到了一幅绘画作品的详细量化特征数据。"
            "请根据这些特征，结合专业的艺术治疗理论，给出一份温暖而深刻的个性化分析。\n\n"
            "=== 分析要求 ===\n\n"
            "1. 【🖼️ 总体印象】(1-2句)\n"
            "用温暖共情的语言描述这幅画给人的第一印象。"
            "例如：'这是一幅笔触轻柔、色调温暖的作品，透露出一种内省和温柔的气质...'\n\n"
            "2. 【📐 空间与构图解读】(2-3条)\n"
            "重点分析画面大小的心理含义（小=克制/细节关注/内向，大=自信/表达欲/需要空间）\n"
            "分析重心位置的心理投射（左右=时间维度(过去vs未来)，上下=精神vs现实维度）\n"
            "指出画面平衡性反映的心理状态（不对称可能反映内心冲突或情绪波动）\n\n"
            "3. 【🎨 色彩情感分析】(2-3条)\n"
            "根据主色调和色彩丰富度推断情绪状态\n"
            "暖色多→情感活跃、可能有焦虑或热情；冷色多→理性克制、可能压抑或沉思\n"
            "暗色比例高→可能正在经历困难情绪或有深度的内在探索\n"
            "单色→专注/克制/可能有情绪压缩；多彩→情感丰富/创造力旺盛\n\n"
            "4. 【💡 深层心理学洞察】(3条)\n"
            "这是最关键的部分。给出真正有深度的洞察：\n"
            "- 可能反映的人格特质或心理需求\n"
            "- 可能暗示的当前心理状态或生命阶段\n"
            "- 画面中可能隐藏的积极资源和力量\n"
            "每条洞察都要结合具体的特征数据，不要泛泛而谈\n\n"
            "5. 【🌱 温和的建议】(1-2条)\n"
            "基于分析，给出一两句温暖的、有实操性的建议。"
            "可以是对绘画的延伸探索邀请，也可以是日常生活中的小练习。\n\n"
            "=== 重要风格要求 ===\n"
            "- 始终保持温暖、共情、非评判的语气\n"
            "- 使用'可能''或许''有时''倾向于'等措辞\n"
            "- 强调'这只是绘画投射的参考视角，不是诊断或标签'\n"
            "- 邀请用户分享：'你觉得这个分析符合你的感受吗？'\n"
            "- 总字数控制在400字以内，每条分析都要精炼有料\n"
            "- 不要列出特征数据本身，而是解读数据背后的心理含义"
        )

        # ---- 构建 user message ----
        user_msg = f"请你分析这幅绘画。以下是详细的量化特征数据：\n\n{feature_desc}\n\n请根据以上数据，结合专业的艺术治疗理论，给出深度分析。"

        # ---- 调用 AI ----
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8,
            max_tokens=800
        )
        analysis_text = response.choices[0].message.content

        return {
            "status": 200,
            "message": "绘画分析完成",
            "data": {
                "analysis": analysis_text,
                "features": features
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"绘画分析失败: {str(e)}\n堆栈: {error_details}")


# 9. 💬 绘画深入交谈 POST 路由
@app.post("/chat_drawing")
async def chat_drawing(payload: dict):
    """
    基于绘画分析结果进行深入对话，
    AI 会记住之前的分析和画面特征，给出更有深度的回应。
    """
    user_message = payload.get("message", "").strip()
    drawing_context = payload.get("drawing_context", {})
    drawing_features = drawing_context.get("features", {})
    drawing_analysis = drawing_context.get("analysis", "")

    if not user_message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    try:
        # 构建对话上下文
        context_summary = ""
        if drawing_features:
            coverage = drawing_features.get('coverage_pct', '未知')
            h_pos = drawing_features.get('horizontal_position', 'center')
            v_pos = drawing_features.get('vertical_position', 'center')
            context_summary = (
                f"这幅画的覆盖率约{coverage}%，"
                f"画面偏{h_pos}偏{v_pos}。"
            )

        # 截取分析摘要（前200字）
        analysis_brief = drawing_analysis[:200] if drawing_analysis else ""

        system_prompt = (
            "你是一位温暖、专业的艺术治疗心理咨询师。\n"
            "用户刚才画了一幅画，你已经给出了初步分析。现在用户想就这幅画进行更深入的交流。\n\n"
            f"【画面信息】: {context_summary}\n"
            f"【之前的分析摘要】: {analysis_brief}\n\n"
            "请像朋友聊天一样，温和、共情地回应用户的问题。\n"
            "鼓励用户分享画背后的故事、感受和联想。\n"
            "使用开放式提问引导用户深入探索自己的内心世界。\n"
            "不要说教，不要绝对化，不要做诊断——我们只是在聊天和探索。"
        )

        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8,
            max_tokens=600
        )

        ai_reply = response.choices[0].message.content

        return {
            "status": 200,
            "message": "发送成功",
            "data": {
                "reply": ai_reply
            }
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}\n堆栈: {error_details}")


# 10. 🛏️ CBT-I 失眠认知行为治疗专用对话路由
@app.post("/chat_cbti")
async def chat_cbti(payload: dict):
    """
    CBT-I 专业失眠指导。
    内置完整的临床CBT-I协议作为系统知识，
    支持睡眠评估、睡眠限制、刺激控制、认知重构、放松训练、睡眠日记。
    """
    user_message = payload.get("message", "").strip()
    # 接收睡眠日记数据（可选）
    sleep_diary = payload.get("sleep_diary", None)

    if not user_message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 构建睡眠日记上下文
    diary_context = ""
    if sleep_diary:
        diary_context = f"""
用户今天的睡眠日记：
- 上床时间: {sleep_diary.get('bed_time', '未记录')}
- 入睡大约用了: {sleep_diary.get('sleep_latency', '未记录')}分钟
- 夜间醒来: {sleep_diary.get('night_wakes', '未记录')}次
- 早上起床时间: {sleep_diary.get('wake_time', '未记录')}
- 实际睡着大约: {sleep_diary.get('total_sleep', '未记录')}小时
- 在床上待了: {sleep_diary.get('time_in_bed', '未记录')}小时
- 睡眠效率: {sleep_diary.get('sleep_efficiency', '未计算')}%
- 日间疲劳度(1-10): {sleep_diary.get('fatigue', '未记录')}
"""

    try:
        system_prompt = """你是一位专门从事CBT-I（失眠认知行为治疗）的临床心理学家，拥有丰富的失眠治疗经验。

=== 你的核心知识库（必须严格基于以下内容回复） ===

【什么是CBT-I】
CBT-I是国际睡眠医学会推荐的慢性失眠一线治疗，由5个核心模块组成，通常6-8周一个疗程。效果持久，优于安眠药。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块1: 睡眠限制疗法 (Sleep Restriction Therapy, SRT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原理: 限制卧床时间，使其接近实际睡眠时间，增加睡眠驱动力（腺苷积累），提高睡眠效率。

操作步骤:
1. 用睡眠日记记录1-2周，计算平均实际睡眠时间(AST)
2. 初始卧床时间(TIB) = AST（最少不低于5小时）
3. 固定起床时间（即使周末也雷打不动）
4. 从起床时间倒推TIB，得到上床时间
   例: AST=6小时，固定7:00起床 → 上床时间=1:00
5. 每周评估睡眠效率(SE = AST/TIB × 100%)
   - SE > 90% → 下周TIB增加15分钟
   - SE 85-90% → 维持不变
   - SE < 85% → 下周TIB减少15分钟
6. 持续调整直到找到个人最优TIB

常见困难与应对:
- "我在床上待那么久都睡不着，你让我更晚上床？" → 解释：这是在帮你重建睡眠驱动力，短期不适换长期好睡眠
- "我害怕睡不够" → 正常反应，头1-2周白天会困，但睡眠质量会快速提升
- 如果半夜醒来睡不着 → 用刺激控制（见模块2）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块2: 刺激控制疗法 (Stimulus Control Therapy, SCT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原理: 重新建立"床=睡觉"的强条件反射，打破"床=焦虑/清醒/玩手机"的联结。

5条黄金法则:
1. 只有困了才上床（不是累了，是眼睛睁不开了的那种困）
2. 床只用来睡觉和性生活。不在床上玩手机、看书、吃东西、想问题
3. 躺下20分钟还睡不着 → 立刻起床，离开卧室，去客厅做无聊的放松的事（听舒缓音乐、看枯燥的书、折纸、数呼吸），等到再次有浓烈困意再回床。重复此过程直到睡着。
4. 如果回床后还是睡不着 → 再次起床，重复步骤3。一晚上可能需要反复3-5次，这是正常的。
5. 无论前一晚睡了多久，每天早上同一时间起床。不补觉、不午睡超20分钟。

关键点:
- "20分钟"不需要精确计时，当你开始焦虑"怎么还睡不着"时就是起床信号
- 起床不是为了惩罚自己，而是打破"在床上痛苦清醒"的恶性循环
- 头几天可能更累，但2周内睡眠效率通常显著提升

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块3: 认知重构 (Cognitive Restructuring)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原理: 失眠者的灾难化思维（"今晚肯定又睡不着""睡不够明天就完蛋了"）本身就是维持失眠的关键因素。改变这些错误信念。

常见失眠认知歪曲及科学修正:

歪曲1: "我必须睡够8小时"
→ 修正: 睡眠需求因人而异(6-9小时)，且每晚不同。有人天生短睡眠者(6小时就够)。关键是白天功能和精神状态，不是钟表上的数字。很多成功人士睡6小时。

歪曲2: "昨晚没睡好，今天肯定废了"
→ 修正: 这叫"睡眠绩效焦虑"。实际上，人体有强大的代偿机制——即使只睡4-5小时，第二天通常能完成大部分任务（可能效率稍有下降，但远非"废了"）。夸大后果本身让你今晚更难入睡。

歪曲3: "躺在床上至少能休息一下"
→ 修正: 在床上清醒地躺几个小时，实际上是在训练你的大脑"在床上清醒是正常的"。躺着休息 ≠ 睡眠。浅睡 ≠ 高质量睡眠恢复。

歪曲4: "我必须要控制睡眠"
→ 修正: 睡眠是生理过程，不是意志行为。越努力越睡不着。你无法强迫自己入睡，但你可以创造有利于睡眠的条件。把任务从"我必须睡着"变成"我只需要躺在床上休息"。

歪曲5: "吃安眠药是唯一的办法"
→ 修正: 安眠药只能短期使用(2-4周)，长期效果下降且可能依赖。CBT-I的效果比安眠药更持久，治愈率60-80%，且无副作用。

认知重构技术 - 三栏表:
| 自动思维 | 支持证据 | 反对证据/替代思维 |
| 今晚肯定睡不着 | 昨晚也没睡好 | 我并非每晚都失眠，上周三就睡得还行；即使睡得少，我也能过完一天 |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块4: 睡眠卫生教育 (Sleep Hygiene)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
行为指南:
1. 咖啡因: 下午2点后不喝咖啡/茶/可乐。咖啡因半衰期4-6小时
2. 酒精: 睡前3小时内不饮酒。酒精虽然让你容易入睡，但会破坏后半夜睡眠结构（REM反弹）
3. 尼古丁: 尼古丁是兴奋剂，睡前一小时不吸烟
4. 运动: 规律运动有助于深睡眠，但睡前2-3小时不做剧烈运动
5. 饮食: 睡前不大量进食，也不饿着肚子上床。小份温牛奶/香蕉/全麦饼干OK
6. 光线: 睡前1-2小时调暗灯光，避免手机/电脑蓝光。早晨起床后立即接触明亮自然光15-30分钟（这是最强的生物钟同步器！）
7. 卧室环境: 安静、黑暗、凉爽(18-22°C最适宜)。用遮光窗帘、白噪音机或耳塞
8. 睡前仪式: 建立固定的放松程序(30-60分钟) — 泡脚、听音乐、轻柔拉伸、冥想、阅读轻松书籍

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块5: 放松训练 (Relaxation Training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
生理放松技术:

A) 渐进式肌肉放松(PMR):
从脚趾开始，逐步往上——
绷紧脚趾5秒 → 突然完全放松 → 感受15秒 → 绷紧小腿5秒 → 放松15秒 → 绷紧大腿 → ... → 腹部 → 胸部 → 肩膀 → 手臂 → 手掌 → 面部 → 前额
全程约15-20分钟。目标：学会区分"紧张"与"放松"的肌肉感觉。

B) 4-7-8呼吸法:
- 用鼻子吸气，默数4秒
- 屏住呼吸，默数7秒
- 用嘴巴缓慢呼气，默数8秒（可以发出轻轻的"嘶"声）
- 重复4轮
生理原理: 长呼气激活副交感神经（迷走神经），降低心率。

C) 腹式呼吸:
一只手放胸口，一只手放腹部。吸气时只让腹部的手升起（胸不动）。慢吸慢呼，每分钟6次左右。睡前做5-10分钟。

D) 引导式想象:
闭上眼睛，想象一个让你感到安全、宁静的地方（如海边、森林、童年小屋）。调动所有感官：看到什么颜色？听到什么声音？闻到什么气味？皮肤感受到什么？越具体越好。

E) 身体扫描:
平躺，注意力从脚趾缓慢移动到头顶。对每个部位：只是觉察感觉（温热、凉、麻、压力），不评价、不改变。如果某个部位紧张，用呼吸把气息"送到"那里，想象紧张随呼气流出。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块6: 睡眠日记 (Sleep Diary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每天早晨记录:
1. 昨天几点上床的?
2. 大概多久睡着?（估计即可，不用看钟）
3. 昨晚醒了几次? 每次大概多久?
4. 今天几点起床的?
5. 从躺下到起床总共几个小时? (Time in Bed)
6. 估计实际睡了几个小时? (Actual Sleep Time)
7. 白天疲劳程度(1-10分)
8. 白天功能受影响程度(1-10分)

计算: 睡眠效率(SE%) = 实际睡眠时间 ÷ 卧床时间 × 100%
目标: SE > 85%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CBT-I 典型疗程安排（6-8周）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第1周: 评估 — 介绍CBT-I，开始记录睡眠日记，睡眠卫生教育
第2周: 睡眠限制开始 — 根据基线数据设定初始TIB，刺激控制5法则
第3-4周: 调整 — 根据SE%调整TIB，引入认知重构
第5-6周: 巩固 — 继续调整TIB，强化认知技能，处理复发预防
第7-8周: 维持 — 回顾进步，制定长期维持计划

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
你的工作方式：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 首次对话：询问用户的睡眠模式，帮助建立基线
2. 每次对话：像教练一样，温和但坚定地引导用户执行CBT-I步骤
3. 询问睡眠日记数据，计算睡眠效率，给出TIB调整建议
4. 当用户表达灾难化思维时，使用认知重构技术温和地挑战
5. 遇到困难时：先共情（"我知道这很难"），再解释原理，最后鼓励坚持
6. 用具体数字说话（"你本周睡眠效率从68%提升到76%，进步很大！"）

【重要】
- 你是在做CBT-I，不是普通聊天。每次对话都要推进治疗进程。
- 主动询问睡眠日记数据，不要等用户自己给。
- 用户可能抗拒睡眠限制（太晚上床），要耐心解释科学原理并鼓励。
- 如果用户有严重抑郁、自杀风险或睡眠呼吸暂停症状（打鼾、呼吸暂停、白天极度嗜睡），建议就医。
- 你不是医生，不做医学诊断。你提供的是心理教育和行为指导。

=== 🔴 交互约束 ===
1.【一次一问】收集睡眠数据时，每次只问一个核心问题。可以先给用户一些共情或简短解释（2-3句），再问问题。
2.【口语化】用口语，像朋友聊微信。
3.【直接给结论】用户问具体建议时，直接给出方案和理由，不用反问。
4.【有温度的专业】每条回复有实质内容，不只是提问。结合CBT-I理论给一些科普小知识。"""

        # 拼接用户消息
        full_message = f"{diary_context}\n用户说: {user_message}" if diary_context else user_message

        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_message}
            ],
            temperature=0.7,
            max_tokens=600
        )

        ai_reply = response.choices[0].message.content

        return {
            "status": 200,
            "message": "CBT-I回复成功",
            "data": {
                "reply": ai_reply
            }
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"CBT-I对话失败: {str(e)}\n堆栈: {error_details}")


# ============================================================
# 11. 🛏️ 睡眠数据 API
# ============================================================

@app.get("/api/sleep/today")
async def sleep_today(request: Request):
    user_id = verify_token(request)
    today = datetime.date.today().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT * FROM sleep_records WHERE user_id = ? AND date = ?"
        cursor.execute(sql, (user_id, today))
        row = cursor.fetchone()
        if not row:
            return ok(None, "今日暂无睡眠数据")
        return ok({
            "date": str(row["date"]),
            "score": row["score"],
            "duration": row["duration"],
            "bedtime": row["bedtime"],
            "waketime": row["waketime"],
            "stages": {
                "deep": row["deep_sleep"],
                "light": row["light_sleep"],
                "rem": row["rem_sleep"],
                "awake": row["awake_time"]
            }
        })
    finally:
        cursor.close()
        conn.close()


@app.get("/api/sleep/detail")
async def sleep_detail(request: Request, date: str = None):
    user_id = verify_token(request)
    if not date:
        date = datetime.date.today().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT * FROM sleep_records WHERE user_id = ? AND date = ?"
        cursor.execute(sql, (user_id, date))
        row = cursor.fetchone()
        if not row:
            return ok(None, "该日期暂无睡眠数据")
        return ok({
            "date": str(row["date"]),
            "score": row["score"],
            "duration": row["duration"],
            "bedtime": row["bedtime"],
            "waketime": row["waketime"],
            "stages": {
                "deep": row["deep_sleep"],
                "light": row["light_sleep"],
                "rem": row["rem_sleep"],
                "awake": row["awake_time"]
            }
        })
    finally:
        cursor.close()
        conn.close()


@app.get("/api/sleep/trend")
async def sleep_trend(request: Request, range: str = "week"):
    user_id = verify_token(request)
    days = 7 if range == "week" else (1 if range == "day" else 30)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """SELECT date, score, duration, deep_sleep, light_sleep, rem_sleep, awake_time
                 FROM sleep_records WHERE user_id = ?
                 ORDER BY date DESC LIMIT ?"""
        cursor.execute(sql, (user_id, days))
        rows = cursor.fetchall()
        result = []
        for row in reversed(rows):
            result.append({
                "date": str(row["date"]),
                "score": row["score"],
                "duration": row["duration"],
                "deepSleep": row["deep_sleep"],
                "lightSleep": row["light_sleep"],
                "rem": row["rem_sleep"],
                "awake": row["awake_time"]
            })
        return ok(result)
    finally:
        cursor.close()
        conn.close()


@app.get("/api/sleep/tips")
async def sleep_tips():
    tips = [
        {"id": "tip_001", "title": "睡前远离电子屏幕", "content": "睡前1小时避免使用手机、电脑和平板。蓝光会抑制褪黑素分泌，影响入睡。试试读一本纸质书或听一些舒缓的音乐来代替刷手机。", "category": "habit", "source": "美国睡眠医学会 (AASM)"},
        {"id": "tip_002", "title": "保持规律作息", "content": "每天在同一时间上床和起床（包括周末！）。固定的作息时间帮助身体建立稳定的生物钟，让你更容易入睡和自然醒来。", "category": "habit", "source": "哈佛医学院睡眠医学部"},
        {"id": "tip_003", "title": "优化卧室环境", "content": "保持卧室安静、黑暗、凉爽（18-22°C最适宜）。使用遮光窗帘、白噪音机或耳塞来减少干扰。舒适的床垫和枕头也至关重要。", "category": "environment", "source": "美国国家睡眠基金会"},
        {"id": "tip_004", "title": "注意饮食与睡眠", "content": "睡前3小时内避免大量进食。下午2点后不喝咖啡。虽然酒精可能让你容易入睡，但它会破坏后半夜的睡眠质量。", "category": "diet", "source": "约翰霍普金斯睡眠中心"},
        {"id": "tip_005", "title": "规律运动助眠", "content": "每周进行150分钟中等强度有氧运动可以显著改善睡眠质量。但睡前2-3小时内避免剧烈运动，以免过度刺激神经系统。", "category": "exercise", "source": "美国心脏协会 (AHA)"}
    ]
    return ok(tips)


@app.post("/api/sleep/record")
async def sleep_record(request: Request, payload: dict):
    user_id = verify_token(request)
    date = payload.get("date", datetime.date.today().isoformat())
    bedtime = payload.get("bedtime", "")
    waketime = payload.get("waketime", "")
    duration = payload.get("duration", 0)
    stages = payload.get("stages", {})
    deep_sleep = stages.get("deep", 0)
    light_sleep = stages.get("light", 0)
    rem_sleep = stages.get("rem", 0)
    awake_time = stages.get("awake", 0)

    total_sleep = deep_sleep + light_sleep + rem_sleep
    score = 0
    if 420 <= total_sleep <= 540:
        score += 40
    elif 360 <= total_sleep <= 600:
        score += 30
    elif total_sleep > 0:
        score += 20
    if total_sleep > 0:
        deep_ratio = deep_sleep / total_sleep
        if 0.15 <= deep_ratio <= 0.30:
            score += 30
        elif 0.10 <= deep_ratio <= 0.35:
            score += 20
        else:
            score += 10
    if duration > 0:
        awake_ratio = awake_time / duration
        if awake_ratio < 0.05:
            score += 30
        elif awake_ratio < 0.10:
            score += 25
        elif awake_ratio < 0.15:
            score += 15
        else:
            score += 5

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """INSERT INTO sleep_records (user_id, date, score, duration, bedtime, waketime, deep_sleep, light_sleep, rem_sleep, awake_time)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON DUPLICATE KEY UPDATE
                 score=VALUES(score), duration=VALUES(duration), bedtime=VALUES(bedtime), waketime=VALUES(waketime),
                 deep_sleep=VALUES(deep_sleep), light_sleep=VALUES(light_sleep), rem_sleep=VALUES(rem_sleep), awake_time=VALUES(awake_time)"""
        cursor.execute(sql, (user_id, date, score, duration, bedtime, waketime, deep_sleep, light_sleep, rem_sleep, awake_time))
        conn.commit()
        return ok({
            "date": date, "score": score, "duration": duration,
            "bedtime": bedtime, "waketime": waketime,
            "stages": {"deep": deep_sleep, "light": light_sleep, "rem": rem_sleep, "awake": awake_time}
        }, "睡眠记录已保存")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 12. 😊 情绪数据 API
# ============================================================

MOOD_VALUE_MAP = {"happy": 5, "calm": 4, "neutral": 3, "sad": 2, "anxious": 1}

@app.get("/api/emotion/today")
async def emotion_today(request: Request):
    user_id = verify_token(request)
    today = datetime.date.today().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT * FROM emotion_records WHERE user_id = ? AND date = ? ORDER BY id DESC LIMIT 1"
        cursor.execute(sql, (user_id, today))
        row = cursor.fetchone()
        if not row:
            return ok(None, "今日暂无打卡")
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else (row["tags"] or [])
        context = json.loads(row["context"]) if isinstance(row["context"], str) else (row["context"] or {})
        return ok({
            "id": f"mood_{row['id']:04d}",
            "date": str(row["date"]),
            "time": row["time"],
            "mood": row["mood"],
            "note": row["note"] or "",
            "tags": tags,
            "context": context
        })
    finally:
        cursor.close()
        conn.close()


@app.get("/api/emotion/monthly")
async def emotion_monthly(request: Request, month: str = None):
    user_id = verify_token(request)
    if not month:
        month = datetime.date.today().strftime("%Y-%m")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """SELECT date, mood, note FROM emotion_records
                 WHERE user_id = ? AND DATE_FORMAT(date, '%%Y-%%m') = ? ORDER BY date"""
        cursor.execute(sql, (user_id, month))
        rows = cursor.fetchall()
        return ok([{
            "date": str(row["date"]),
            "mood": row["mood"],
            "hasNote": bool(row["note"] and row["note"].strip())
        } for row in rows])
    finally:
        cursor.close()
        conn.close()


@app.get("/api/emotion/stats")
async def emotion_stats(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""SELECT mood, COUNT(*) as cnt FROM emotion_records
                          WHERE user_id = ? AND DATE_FORMAT(date, '%%Y-%%m') = DATE_FORMAT(CURDATE(), '%%Y-%%m')
                          GROUP BY mood""", (user_id,))
        rows = cursor.fetchall()
        stats = {"happy": 0, "calm": 0, "neutral": 0, "sad": 0, "anxious": 0, "total": 0}
        for row in rows:
            mood = row["mood"]
            if mood in stats:
                stats[mood] = row["cnt"]
                stats["total"] += row["cnt"]
        return ok(stats)
    finally:
        cursor.close()
        conn.close()


@app.get("/api/emotion/trend")
async def emotion_trend(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT date, mood FROM emotion_records WHERE user_id = ? ORDER BY date DESC LIMIT 30"
        cursor.execute(sql, (user_id,))
        rows = cursor.fetchall()
        result = []
        for row in reversed(rows):
            result.append({
                "date": str(row["date"]),
                "mood": row["mood"],
                "moodValue": MOOD_VALUE_MAP.get(row["mood"], 3),
                "note": ""
            })
        return ok(result)
    finally:
        cursor.close()
        conn.close()


@app.get("/api/emotion/notes")
async def emotion_notes(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """SELECT id, date, time, mood, note, tags FROM emotion_records
                 WHERE user_id = ? AND note IS NOT NULL AND note != ''
                 ORDER BY date DESC, id DESC LIMIT 5"""
        cursor.execute(sql, (user_id,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else (row["tags"] or [])
            result.append({
                "id": f"note_{row['id']:04d}",
                "date": str(row["date"]),
                "time": row["time"],
                "mood": row["mood"],
                "content": row["note"],
                "tags": tags
            })
        return ok(result)
    finally:
        cursor.close()
        conn.close()


@app.get("/api/emotion/streak")
async def emotion_streak(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "SELECT DISTINCT date FROM emotion_records WHERE user_id = ? ORDER BY date DESC"
        cursor.execute(sql, (user_id,))
        rows = cursor.fetchall()
        if not rows:
            return ok({"streak": 0})
        today = datetime.date.today()
        streak = 0
        check_date = today
        dates_set = {str(row["date"]) for row in rows}
        while check_date.isoformat() in dates_set:
            streak += 1
            check_date = check_date - datetime.timedelta(days=1)
        if today.isoformat() not in dates_set and streak == 0:
            yesterday = today - datetime.timedelta(days=1)
            check_date = yesterday
            while check_date.isoformat() in dates_set:
                streak += 1
                check_date = check_date - datetime.timedelta(days=1)
        return ok({"streak": streak})
    finally:
        cursor.close()
        conn.close()


@app.post("/api/emotion/checkin")
async def emotion_checkin(request: Request, payload: dict):
    user_id = verify_token(request)
    mood = payload.get("mood", "")
    note = payload.get("note", "")
    tags = payload.get("tags", [])
    context = payload.get("context", {})
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """INSERT INTO emotion_records (user_id, date, time, mood, note, tags, context)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sql, (user_id, today, now_time, mood, note, json.dumps(tags, ensure_ascii=False), json.dumps(context, ensure_ascii=False)))
        conn.commit()
        new_id = cursor.lastrowid
        return ok({
            "id": f"mood_{new_id:04d}",
            "date": today,
            "time": now_time,
            "mood": mood,
            "note": note,
            "tags": tags,
            "context": context
        }, "打卡成功")
    finally:
        cursor.close()
        conn.close()


@app.delete("/api/emotion/notes/{note_id}")
async def emotion_delete_note(request: Request, note_id: str):
    user_id = verify_token(request)
    try:
        numeric_id = int(note_id.split("_")[1])
    except (IndexError, ValueError):
        raise HTTPException(status_code=400, detail="无效的笔记ID格式")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM emotion_records WHERE id = ? AND user_id = ?", (numeric_id, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="笔记不存在或无权删除")
        conn.commit()
        return ok(True, "已删除")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 13. ⏰ 闹钟 API
# ============================================================

def _parse_alarm_id(alarm_id: str) -> int:
    try:
        return int(alarm_id.split("_")[1]) if "_" in alarm_id else int(alarm_id)
    except (IndexError, ValueError):
        raise HTTPException(status_code=400, detail="无效的闹钟ID")


@app.get("/api/alarms")
async def alarm_list(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM alarms WHERE user_id = ? ORDER BY time", (user_id,))
        rows = cursor.fetchall()
        return ok([{
            "id": f"alarm_{row['id']:04d}",
            "time": row["time"],
            "label": row["label"],
            "repeat": row["repeat_type"],
            "smartWake": bool(row["smart_wake"]),
            "ringtone": row["ringtone"],
            "enabled": bool(row["enabled"]),
            "type": row["type"]
        } for row in rows])
    finally:
        cursor.close()
        conn.close()


@app.post("/api/alarms")
async def alarm_create(request: Request, payload: dict):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """INSERT INTO alarms (user_id, time, label, repeat_type, smart_wake, ringtone, enabled, type)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sql, (
            user_id,
            payload.get("time", "07:00"),
            payload.get("label", "闹钟"),
            payload.get("repeat", "weekdays"),
            1 if payload.get("smartWake") else 0,
            payload.get("ringtone", "默认"),
            1 if payload.get("enabled", True) else 0,
            payload.get("type", "wakeup")
        ))
        conn.commit()
        new_id = cursor.lastrowid
        return ok({
            "id": f"alarm_{new_id:04d}",
            "time": payload.get("time", "07:00"),
            "label": payload.get("label", "闹钟"),
            "repeat": payload.get("repeat", "weekdays"),
            "smartWake": payload.get("smartWake", False),
            "ringtone": payload.get("ringtone", "默认"),
            "enabled": payload.get("enabled", True),
            "type": payload.get("type", "wakeup")
        }, "闹钟已创建")
    finally:
        cursor.close()
        conn.close()


@app.put("/api/alarms/{alarm_id}")
async def alarm_update(request: Request, alarm_id: str, payload: dict):
    user_id = verify_token(request)
    numeric_id = _parse_alarm_id(alarm_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """UPDATE alarms SET time=?, label=?, repeat_type=?, smart_wake=?, ringtone=?, enabled=?, type=?
                 WHERE id = ? AND user_id = ?"""
        cursor.execute(sql, (
            payload.get("time", "07:00"),
            payload.get("label", "闹钟"),
            payload.get("repeat", "weekdays"),
            1 if payload.get("smartWake") else 0,
            payload.get("ringtone", "默认"),
            1 if payload.get("enabled", True) else 0,
            payload.get("type", "wakeup"),
            numeric_id,
            user_id
        ))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="闹钟不存在")
        conn.commit()
        return ok({**payload, "id": alarm_id}, "闹钟已更新")
    finally:
        cursor.close()
        conn.close()


@app.put("/api/alarms/{alarm_id}/toggle")
async def alarm_toggle(request: Request, alarm_id: str):
    user_id = verify_token(request)
    numeric_id = _parse_alarm_id(alarm_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE alarms SET enabled = NOT enabled WHERE id = ? AND user_id = ?", (numeric_id, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="闹钟不存在")
        conn.commit()
        return ok(True, "已切换")
    finally:
        cursor.close()
        conn.close()


@app.delete("/api/alarms/{alarm_id}")
async def alarm_delete(request: Request, alarm_id: str):
    user_id = verify_token(request)
    numeric_id = _parse_alarm_id(alarm_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM alarms WHERE id = ? AND user_id = ?", (numeric_id, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="闹钟不存在")
        conn.commit()
        return ok(True, "已删除")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 14. 📱 设备绑定 API
# ============================================================

def _parse_device_id(device_id: str) -> int:
    try:
        return int(device_id.split("_")[1]) if "_" in device_id else int(device_id)
    except (IndexError, ValueError):
        raise HTTPException(status_code=400, detail="无效的设备ID")


@app.get("/api/devices")
async def device_list(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM devices WHERE user_id = ? ORDER BY id", (user_id,))
        rows = cursor.fetchall()
        return ok([{
            "id": f"dev_{row['id']:04d}",
            "type": row["device_type"],
            "name": row["name"],
            "brand": row["brand"],
            "battery": row["battery"],
            "connectionStatus": row["connection_status"],
            "lastSync": row["last_sync"] or "从未同步"
        } for row in rows])
    finally:
        cursor.close()
        conn.close()


@app.post("/api/devices/bind")
async def device_bind(request: Request, payload: dict):
    user_id = verify_token(request)
    device_id = payload.get("deviceId", "")
    if not device_id:
        raise HTTPException(status_code=400, detail="设备ID不能为空")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """INSERT INTO devices (user_id, device_type, name, brand, battery, connection_status, last_sync)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sql, (
            user_id,
            payload.get("type", "watch"),
            payload.get("name", f"设备 {device_id}"),
            payload.get("brand", "未知品牌"),
            payload.get("battery", 100),
            "connected",
            "刚刚同步"
        ))
        conn.commit()
        new_id = cursor.lastrowid
        return ok({
            "id": f"dev_{new_id:04d}",
            "type": payload.get("type", "watch"),
            "name": payload.get("name", f"设备 {device_id}"),
            "brand": payload.get("brand", "未知品牌"),
            "battery": payload.get("battery", 100),
            "connectionStatus": "connected",
            "lastSync": "刚刚同步"
        }, "设备绑定成功")
    finally:
        cursor.close()
        conn.close()


@app.delete("/api/devices/{device_id}")
async def device_unbind(request: Request, device_id: str):
    user_id = verify_token(request)
    numeric_id = _parse_device_id(device_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM devices WHERE id = ? AND user_id = ?", (numeric_id, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="设备不存在")
        conn.commit()
        return ok(True, "已解绑")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 15. ⚙️ 用户设置 API
# ============================================================

@app.get("/api/settings")
async def settings_get(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            cursor.execute("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
            conn.commit()
            cursor.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
        return ok({
            "theme": row["theme"],
            "whiteNoise": row["white_noise"],
            "sleepReminderEnabled": bool(row["sleep_reminder"]),
            "smartWakeEnabled": bool(row["smart_wake"]),
            "snoreDetectionEnabled": bool(row["snore_detection"]),
            "notificationEnabled": bool(row["notification"]),
            "bedtimeReminderEnabled": bool(row["bedtime_reminder"])
        })
    finally:
        cursor.close()
        conn.close()


@app.put("/api/settings")
async def settings_update(request: Request, payload: dict):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM user_settings WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            cursor.execute("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
            conn.commit()
        updates = []
        values = []
        field_map = {
            "theme": "theme",
            "whiteNoise": "white_noise",
            "sleepReminderEnabled": "sleep_reminder",
            "smartWakeEnabled": "smart_wake",
            "snoreDetectionEnabled": "snore_detection",
            "notificationEnabled": "notification",
            "bedtimeReminderEnabled": "bedtime_reminder"
        }
        for json_key, db_col in field_map.items():
            if json_key in payload:
                val = payload[json_key]
                updates.append(f"{db_col} = ?")
                values.append(1 if isinstance(val, bool) and val else (0 if isinstance(val, bool) and not val else val))
        if updates:
            values.append(user_id)
            sql = f"UPDATE user_settings SET {', '.join(updates)} WHERE user_id = ?"
            cursor.execute(sql, values)
            conn.commit()
        return ok(payload, "设置已更新")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 16. 👤 用户资料 API
# ============================================================

@app.get("/api/user/profile")
async def user_profile(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM user_account WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return ok({
            "id": f"u_{user['id']:04d}",
            "nickname": user["nickname"],
            "phone": user["phone"] or "",
            "avatar": user["avatar"],
            "gender": user["gender"],
            "bio": user["sign"] or "",
            "age": user.get("age", 0) if isinstance(user, dict) else 0,
            "memberLevel": "premium",
            "sleepGoal": {
                "targetDuration": 480,
                "targetBedtime": "23:00",
                "targetWaketime": "07:00",
                "alarmEnabled": True
            }
        })
    finally:
        cursor.close()
        conn.close()

# ============================================================
# 17. 💬 对话历史 API
# ============================================================

@app.get("/api/conversations")
async def conversation_list(request: Request):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """SELECT id, title, message_count, created_at, updated_at
                 FROM conversations WHERE user_id = ? ORDER BY updated_at DESC"""
        cursor.execute(sql, (user_id,))
        rows = cursor.fetchall()
        return ok([{
            "id": row["id"],
            "title": row["title"],
            "messageCount": row["message_count"],
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"])
        } for row in rows])
    finally:
        cursor.close()
        conn.close()


@app.post("/api/conversations")
async def conversation_create(request: Request, payload: dict):
    user_id = verify_token(request)
    title = payload.get("title", "新对话")[:200]
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )
        conn.commit()
        new_id = cursor.lastrowid
        return ok({
            "id": new_id,
            "title": title,
            "messageCount": 0,
            "createdAt": str(datetime.datetime.now()),
            "updatedAt": str(datetime.datetime.now())
        }, "对话已创建")
    finally:
        cursor.close()
        conn.close()


@app.get("/api/conversations/{conv_id}")
async def conversation_detail(request: Request, conv_id: int):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, title FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
        conv = cursor.fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="对话不存在")
        cursor.execute(
            "SELECT id, role, content, source, emotion, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,)
        )
        msgs = cursor.fetchall()
        return ok({
            "id": conv["id"],
            "title": conv["title"],
            "messages": [{
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "source": m["source"] or "",
                "emotion": json.loads(m["emotion"]) if isinstance(m["emotion"], str) else (m["emotion"] or None),
                "timestamp": str(m["created_at"])
            } for m in msgs]
        })
    finally:
        cursor.close()
        conn.close()


@app.delete("/api/conversations/{conv_id}")
async def conversation_delete(request: Request, conv_id: int):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="对话不存在")
        conn.commit()
        return ok(True, "已删除")
    finally:
        cursor.close()
        conn.close()


@app.post("/api/conversations/{conv_id}/messages")
async def conversation_add_messages(request: Request, conv_id: int, payload: dict):
    user_id = verify_token(request)
    msgs = payload.get("messages", [])
    if not msgs:
        raise HTTPException(status_code=400, detail="消息列表不能为空")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="对话不存在")
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            source = m.get("source", "")
            emotion = m.get("emotion", None)
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, source, emotion) VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, content, source, json.dumps(emotion, ensure_ascii=False) if emotion else None)
            )
        new_count = len(msgs)
        cursor.execute(
            "UPDATE conversations SET message_count = message_count + ?, updated_at = NOW() WHERE id = ?",
            (new_count, conv_id)
        )
        conn.commit()
        return ok({"added": new_count}, "消息已保存")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 18. 👤 用户注册 + 头像 + 资料更新 API
# ============================================================

@app.post("/register")
async def user_register(payload: dict):
    username = payload.get("username", "").strip()
    password = payload.get("password", "").strip()
    phone = payload.get("phone", "").strip()

    if not username or len(username) < 2:
        raise HTTPException(status_code=400, detail="账号至少需要2个字符")
    if not password or len(password) < 4:
        raise HTTPException(status_code=400, detail="密码至少需要4个字符")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM user_account WHERE username = ?", (username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="该账号已被注册，请换一个试试")

        # 创建用户
        default_nickname = username
        default_avatar = "/static/7.jpeg"
        cursor.execute(
            "INSERT INTO user_account (username, password, phone, nickname, avatar, gender, sign) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, password, phone or "", default_nickname, default_avatar, "保密", "")
        )
        conn.commit()
        user_id = cursor.lastrowid

        # 生成 JWT
        jwt_payload = {
            "user_id": user_id,
            "nickname": default_nickname,
            "tel": phone or "",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        token = jwt.encode(jwt_payload, SECRET_KEY, algorithm=ALGORITHM)

        return {
            "status": 200,
            "message": "注册成功",
            "data100": {
                "token": token,
                "user": {
                    "id": user_id,
                    "nickname": default_nickname,
                    "tel": phone or "",
                    "gender": "保密",
                    "bio": "",
                    "avatar": default_avatar
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"注册失败: {repr(e)}")
    finally:
        cursor.close()
        conn.close()


@app.put("/api/user/avatar")
async def user_update_avatar(request: Request, payload: dict):
    user_id = verify_token(request)
    image_b64 = payload.get("image", "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="图片数据不能为空")

    try:
        # 解码 base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)

        # 保存到 static/avatars/
        avatar_dir = "static/avatars"
        os.makedirs(avatar_dir, exist_ok=True)
        filename = f"avatar_{user_id}_{int(datetime.datetime.now().timestamp())}.jpg"
        filepath = os.path.join(avatar_dir, filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        avatar_url = f"/{filepath.replace(os.sep, '/')}"

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE user_account SET avatar = ? WHERE id = ?", (avatar_url, user_id))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

        return ok({"avatar": avatar_url}, "头像更新成功")
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"头像上传失败: {repr(e)}")


@app.put("/api/user/profile")
async def user_update_profile(request: Request, payload: dict):
    user_id = verify_token(request)
    nickname = payload.get("nickname")
    phone = payload.get("phone")
    gender = payload.get("gender")
    bio = payload.get("bio")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        updates = []
        values = []
        if nickname is not None:
            updates.append("nickname = ?")
            values.append(nickname)
        if phone is not None:
            updates.append("phone = ?")
            values.append(phone)
        if gender is not None:
            updates.append("gender = ?")
            values.append(gender)
        if bio is not None:
            updates.append("sign = ?")
            values.append(bio)

        if updates:
            values.append(user_id)
            sql = f"UPDATE user_account SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(sql, values)
            conn.commit()

        # 返回更新后的用户信息
        cursor.execute("SELECT * FROM user_account WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        return ok({
            "id": user["id"],
            "nickname": user["nickname"],
            "phone": user["phone"],
            "gender": user["gender"],
            "bio": user["sign"] or "",
            "avatar": user["avatar"]
        }, "资料已更新")
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 19. 📋 聚合记录时间线 API
# ============================================================

@app.get("/api/records/timeline")
async def records_timeline(request: Request, type: str = "all"):
    user_id = verify_token(request)
    conn = get_db_connection()
    cursor = conn.cursor()
    records = []

    try:
        # 1. 对话记录
        if type in ("all", "chat"):
            cursor.execute(
                """SELECT id, title, message_count, created_at, updated_at
                   FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 30""",
                (user_id,)
            )
            for row in cursor.fetchall():
                records.append({
                    "id": f"chat_{row['id']}",
                    "type": "chat",
                    "title": row["title"],
                    "subtitle": f"{row['message_count']}条消息",
                    "time": str(row["updated_at"]),
                    "date": str(row["created_at"])[:10],
                    "detail": {"conversationId": row["id"], "messageCount": row["message_count"]}
                })

        # 2. 情绪打卡
        if type in ("all", "emotion"):
            cursor.execute(
                """SELECT id, date, time, mood, note, tags
                   FROM emotion_records WHERE user_id = ? ORDER BY date DESC, id DESC LIMIT 30""",
                (user_id,)
            )
            for row in cursor.fetchall():
                tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else (row["tags"] or [])
                mood_labels = {"happy":"😊 开心", "calm":"😌 平静", "neutral":"😐 一般", "sad":"😢 低落", "anxious":"😰 焦虑"}
                records.append({
                    "id": f"emotion_{row['id']}",
                    "type": "emotion",
                    "title": mood_labels.get(row["mood"], row["mood"]),
                    "subtitle": row["note"][:50] if row["note"] else "仅打卡",
                    "time": row["time"] or "",
                    "date": str(row["date"]),
                    "detail": {"mood": row["mood"], "note": row["note"] or "", "tags": tags}
                })

        # 3. 绘画分析（如果有记录的话暂从 conversations 里取 title 含"绘画"的）
        if type in ("all", "draw"):
            cursor.execute(
                """SELECT id, title, created_at FROM conversations
                   WHERE user_id = ? AND title LIKE ? ORDER BY created_at DESC LIMIT 10""",
                (user_id, "%绘画%")
            )
            for row in cursor.fetchall():
                records.append({
                    "id": f"draw_{row['id']}",
                    "type": "draw",
                    "title": row["title"],
                    "subtitle": "绘画心理分析",
                    "time": str(row["created_at"]),
                    "date": str(row["created_at"])[:10],
                    "detail": {"conversationId": row["id"]}
                })

        # 按日期+时间倒序排列
        records.sort(key=lambda r: r["date"] + (r.get("time") or "00:00"), reverse=True)

        # 按日期分组
        grouped = {}
        for r in records:
            d = r["date"]
            if d not in grouped:
                grouped[d] = {"date": d, "items": []}
            grouped[d]["items"].append(r)

        result = list(grouped.values())
        result.sort(key=lambda g: g["date"], reverse=True)

        return ok(result)
    finally:
        cursor.close()
        conn.close()
