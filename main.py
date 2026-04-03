import os
import io
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import numpy as np

app = FastAPI()

# 允許跨域請求 (React Native 開發必備)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. 初始化 Firebase (支援環境變數與本地檔案)
# ---------------------------------------------------------
# --- 修改後的 Firebase 初始化區塊 ---
try:
    # 1. 優先嘗試從 Render 的環境變數讀取
    firebase_config_env = os.environ.get('FIREBASE_CONFIG')
    
    if firebase_config_env:
        print("✅ 偵測到環境變數，正在初始化 Firebase...")
        # 將環境變數中的字串轉為 JSON 字典
        cred_dict = json.loads(firebase_config_env)
        cred = credentials.Certificate(cred_dict)
    else:
        # 2. 如果沒有環境變數，才找本地檔案 (這是你在電腦測試時用的)
        print("ℹ️ 未找到環境變數，嘗試讀取本地 serviceAccountKey.json...")
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            raise Exception("❌ 找不到任何 Firebase 憑證（環境變數或本地檔案）")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("🚀 Firebase Admin SDK 初始化成功")
except Exception as e:
    print(f"❌ Firebase 初始化失敗: {e}")
    # 在雲端環境如果初始化失敗，直接拋出錯誤讓部署停止，方便除錯
    if os.environ.get('FIREBASE_CONFIG'):
        raise e
# ---------------------------------------------------------
# 2. 初始化 AI 模型 (MobileNet V2)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用權限較新的 weights 參數以避免 Warning
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Identity() 
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(input_data, is_url=True):
    """將圖片轉換為特徵向量"""
    try:
        if is_url:
            response = requests.get(input_data, timeout=10) # 雲端連線稍微加長 timeout
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(io.BytesIO(input_data)).convert('RGB')
        
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = model(tensor).cpu().numpy().flatten()
        return vector
    except Exception as e:
        print(f"向量提取出錯: {e}")
        return None

# ---------------------------------------------------------
# 3. 核心 API: 全自動後端比對
# ---------------------------------------------------------
@app.post("/auto_compare")
async def auto_compare(
    user_id: str = Query(...), 
    mode: str = Query("personal"),
    file: UploadFile = File(...)
):
    try:
        # A. 決定查詢目標
        target_id = user_id
        id_field = "userId"

        if mode == "family":
            family_refs = db.collection("family_members").where("userId", "==", user_id).limit(1).get()
            if family_refs:
                target_id = family_refs[0].to_dict().get("familyId")
                id_field = "familyId"

        # B. 抓取庫存清單 (僅抓取該用戶的產品)
        products_ref = db.collection("products").where(id_field, "==", target_id).stream()
        
        # C. 處理手機拍的照片
        image_content = await file.read()
        current_vector = get_vector(image_content, is_url=False)
        
        if current_vector is None:
            raise HTTPException(status_code=400, detail="無法辨識圖片特徵")

        best_match = None
        highest_score = 0
        threshold = 0.70 # 你測試成功的門檻

        # D. 後端內部比對 (不再需要手機多次傳輸)
        for doc in products_ref:
            p_data = doc.to_dict()
            base_image_url = p_data.get("image")
            
            if not base_image_url or not base_image_url.startswith("http"):
                continue
                
            base_vector = get_vector(base_image_url, is_url=True)
            if base_vector is None: continue

            # 計算 Cosine Similarity
            similarity = np.dot(current_vector, base_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(base_vector))
            
            if similarity > highest_score:
                highest_score = float(similarity)
                best_match = {
                    "id": doc.id,
                    "name": p_data.get("name"),
                    "stock": p_data.get("stock", 0),
                    "score": highest_score,
                    "match": bool(highest_score >= threshold)
                }

        # E. 回傳結果
        if best_match and best_match["match"]:
            return {"status": "success", "data": best_match}
        else:
            return {
                "status": "not_found", 
                "message": "找不到匹配物品", 
                "best_guess": best_match["name"] if best_match else "未知",
                "highest_score": highest_score
            }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "running", "message": "Zenkurenaido AI Backend is online"}

if __name__ == "__main__":
    import uvicorn
    # Render 會自動指定 PORT，本地則預設 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)