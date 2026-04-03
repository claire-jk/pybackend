import os
import io
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

# 1. 初始化 Firebase
# 注意：部署到雲端時，建議將路徑改為環境變數讀取
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# 2. 初始化 AI 模型 (MobileNet V2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Identity() # 移除分類層，只提取特徵向量
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
            response = requests.get(input_data, timeout=5)
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

# --- 新增：全自動比對接口 ---
@app.post("/auto_compare")
async def auto_compare(
    user_id: str = Query(...), 
    mode: str = Query("personal"),
    file: UploadFile = File(...)
):
    """
    接收一張圖片，自動對比該用戶/家庭的所有庫存，回傳最像的結果。
    """
    try:
        # 1. 決定查詢目標 (個人或家庭)
        target_id = user_id
        id_field = "userId"

        if mode == "family":
            family_ref = db.collection("family_members").where("userId", "==", user_id).limit(1).get()
            if family_ref:
                target_id = family_ref[0].to_dict().get("familyId")
                id_field = "familyId"

        # 2. 從 Firestore 抓取該用戶所有「有照片」的產品
        products_ref = db.collection("products").where(id_field, "==", target_id).stream()
        
        # 3. 處理上傳的圖片
        image_content = await file.read()
        current_vector = get_vector(image_content, is_url=False)
        
        if current_vector is None:
            raise HTTPException(status_code=400, detail="無法辨識上傳的圖片內容")

        best_match = None
        highest_score = 0
        threshold = 0.70  # 你之前測試成功的門檻

        # 4. 在後端進行循環比對
        for doc in products_ref:
            p_data = doc.to_dict()
            base_image_url = p_data.get("image")
            
            if not base_image_url:
                continue
                
            base_vector = get_vector(base_image_url, is_url=True)
            if base_vector is None: continue

            # 計算餘弦相似度
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

        # 5. 回傳結果
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
    uvicorn.run(app, host="0.0.0.0", port=8000)