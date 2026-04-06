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
import uvicorn

app = FastAPI(title="Zenkurenaido AI Backend")

# 允許跨域請求 (React Native 開發與本地瀏覽器測試必備)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. 初始化 Firebase
# ---------------------------------------------------------
try:
    firebase_config_env = os.environ.get('FIREBASE_CONFIG')
    if firebase_config_env:
        print("✅ 偵測到環境變數，正在初始化 Firebase...")
        cred_dict = json.loads(firebase_config_env)
        cred = credentials.Certificate(cred_dict)
    else:
        print("ℹ️ 未找到環境變數，嘗試讀取本地 serviceAccountKey.json...")
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            print("⚠️ 警告：找不到 Firebase 憑證，部分功能將無法運作")
            cred = None

    if cred:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("🚀 Firebase Admin SDK 初始化成功")
except Exception as e:
    print(f"❌ Firebase 初始化失敗: {e}")

# ---------------------------------------------------------
# 2. 初始化 AI 模型 (MobileNet V2)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            response = requests.get(input_data, timeout=10)
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            # 處理直接傳入的 bytes 數據
            img = Image.open(io.BytesIO(input_data)).convert('RGB')
        
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = model(tensor).cpu().numpy().flatten()
        return vector
    except Exception as e:
        print(f"向量提取出錯: {e}")
        return None

# ---------------------------------------------------------
# 3. 核心比對邏輯 (工具函數)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. 核心比對邏輯 (更新版：加入終端機 Log)
# ---------------------------------------------------------
def find_best_match_in_db(current_vector, products_ref, threshold=0.70):
    best_match = None
    highest_score = 0
    
    # 這裡增加一個強制輸出的測試
    print("\n" + "="*50, flush=True)
    print("🚀 開始進入 AI 比對邏輯...", flush=True)
    print(f"{'物品名稱':<20} | {'相似度得分':<10}", flush=True)
    print("-"*50, flush=True)

    # 將 stream 轉成 list 確保裡面真的有東西，並 debug 數量
    products_list = list(products_ref)
    print(f"DEBUG: 數據庫中找到 {len(products_list)} 個候選物品進行比對", flush=True)

    for doc in products_list:
        p_data = doc.to_dict()
        name = p_data.get("name", "未命名物品")
        base_image_url = p_data.get("image")
        
        if not base_image_url or not base_image_url.startswith("http"):
            print(f"{name:<20} | ⚠️ 跳過 (無有效圖片連結)", flush=True)
            continue
            
        base_vector = get_vector(base_image_url, is_url=True)
        if base_vector is None: 
            print(f"{name:<20} | ❌ 錯誤 (圖片向量化失敗)", flush=True)
            continue

        # 計算 Cosine Similarity
        similarity = np.dot(current_vector, base_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(base_vector))
        
        # 輸出到終端機
        status_icon = "✅" if similarity >= threshold else "☁️"
        print(f"{name:<20} | {similarity:.4f} {status_icon}", flush=True)

        if similarity > highest_score:
            highest_score = float(similarity)
            best_match = {
                "id": doc.id,
                "name": name,
                "stock": p_data.get("stock", 0),
                "image": base_image_url,
                "score": highest_score,
                "match": bool(highest_score >= threshold)
            }
            
    print("-"*50, flush=True)
    if best_match:
        print(f"🏆 最終比對結果: {best_match['name']} ({best_match['score']:.4f})", flush=True)
    else:
        print("🔍 最終比對結果: 沒找到任何匹配", flush=True)
    print("="*50 + "\n", flush=True)
    
    return best_match

# ---------------------------------------------------------
# 4. 功能 1 & 2：影像辨識 (相機/相簿)
# ---------------------------------------------------------
@app.post("/identify_by_image")
async def identify_by_image(
    user_id: str = Query(...), 
    mode: str = Query("personal"),
    file: UploadFile = File(...)
):
    try:
        # A. 決定查詢對象
        target_id = user_id
        id_field = "userId"
        if mode == "family":
            family_refs = db.collection("family_members").where("userId", "==", user_id).limit(1).get()
            if family_refs:
                target_id = family_refs[0].to_dict().get("familyId")
                id_field = "familyId"

        # B. 取得比對清單
        products_ref = db.collection("products").where(id_field, "==", target_id).stream()
        
        # C. 處理上傳檔案
        image_content = await file.read()
        current_vector = get_vector(image_content, is_url=False)
        
        if current_vector is None:
            raise HTTPException(status_code=400, detail="圖片損壞或格式不支援")

        # D. 比對結果
        result = find_best_match_in_db(current_vector, products_ref)

        if result and result["match"]:
            return {"status": "success", "source": "image", "data": result}
        else:
            return {
                "status": "not_found", 
                "message": "找不到匹配物品", 
                "best_guess": result["name"] if result else "無",
                "score": result["score"] if result else 0
            }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 5. 功能 3：名稱搜尋
# ---------------------------------------------------------
@app.get("/identify_by_name")
async def identify_by_name(
    user_id: str = Query(...),
    mode: str = Query("personal"),
    query_name: str = Query(...)
):
    try:
        target_id = user_id
        id_field = "userId"
        if mode == "family":
            family_refs = db.collection("family_members").where("userId", "==", user_id).limit(1).get()
            if family_refs:
                target_id = family_refs[0].to_dict().get("familyId")
                id_field = "familyId"

        products_ref = db.collection("products").where(id_field, "==", target_id).stream()
        
        matches = []
        for doc in products_ref:
            p_data = doc.to_dict()
            name = p_data.get("name", "")
            # 簡單包含判斷 (不分大小寫)
            if query_name.lower() in name.lower():
                matches.append({
                    "id": doc.id,
                    "name": name,
                    "stock": p_data.get("stock", 0),
                    "image": p_data.get("image")
                })

        if matches:
            return {"status": "success", "source": "text", "count": len(matches), "data": matches}
        else:
            return {"status": "not_found", "message": f"找不到名稱包含 '{query_name}' 的物品"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "online", "description": "Zenkurenaido AI API (Image & Text Search)"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)