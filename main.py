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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Firebase 初始化
# ---------------------------------------------------------
try:
    firebase_config_env = os.environ.get('FIREBASE_CONFIG')
    if firebase_config_env:
        cred_dict = json.loads(firebase_config_env)
        cred = credentials.Certificate(cred_dict)
    else:
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            cred = None

    if cred:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("🚀 Firebase 初始化成功")
except Exception as e:
    print(f"❌ Firebase 初始化失敗: {e}")

# ---------------------------------------------------------
# AI 模型
# ---------------------------------------------------------
torch.set_num_threads(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Identity()
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

vector_cache = {}

def get_vector(input_data, is_url=True):
    try:
        if is_url:
            if input_data in vector_cache:
                return vector_cache[input_data]

            response = requests.get(input_data, timeout=5)
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(io.BytesIO(input_data)).convert('RGB')

        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            vector = model(tensor).cpu().numpy().flatten()

        if is_url:
            vector_cache[input_data] = vector

        return vector

    except Exception as e:
        print(f"向量提取錯誤: {e}")
        return None


# ---------------------------------------------------------
# 🔥 統一 response 格式（關鍵修正）
# ---------------------------------------------------------
def format_result(doc_id, p_data, score):
    return {
        "id": doc_id,
        "name": p_data.get("name", ""),
        "stock": int(p_data.get("stock", 0)),
        "image": p_data.get("image", ""),
        "score": float(score),
        "match": True,
        "location": p_data.get("location", "未設定")
    }


# ---------------------------------------------------------
# 🔥 比對
# ---------------------------------------------------------
def find_best_match_in_db(current_vector, products_ref, threshold=0.70):
    best_match = None
    highest_score = 0

    products_list = list(products_ref)
    print(f"⚡ 比對 {len(products_list)} 個物品")

    for doc in products_list:
        p_data = doc.to_dict()

        base_vector = p_data.get("vector")

        if base_vector:
            base_vector = np.array(base_vector)
        else:
            image_url = p_data.get("image")
            if not image_url:
                continue

            base_vector = get_vector(image_url, is_url=True)
            if base_vector is None:
                continue

            try:
                doc.reference.update({
                    "vector": base_vector.tolist()
                })
            except:
                pass

        similarity = np.dot(current_vector, base_vector) / (
            np.linalg.norm(current_vector) * np.linalg.norm(base_vector)
        )

        if similarity > highest_score:
            highest_score = float(similarity)
            best_match = (doc.id, p_data, similarity)

    return best_match


# ---------------------------------------------------------
# 📷 影像辨識 API（修正版）
# ---------------------------------------------------------
@app.post("/identify_by_image")
async def identify_by_image(
    user_id: str = Query(...),
    mode: str = Query("personal"),
    file: UploadFile = File(...)
):
    try:
        target_id = user_id
        id_field = "userId"

        if mode == "family":
            family_refs = db.collection("family_members").where(
                "userId", "==", user_id
            ).limit(1).get()

            if family_refs:
                target_id = family_refs[0].to_dict().get("familyId")
                id_field = "familyId"

        products_ref = db.collection("products") \
            .where(id_field, "==", target_id) \
            .limit(30) \
            .stream()

        image_content = await file.read()
        current_vector = get_vector(image_content, is_url=False)

        if current_vector is None:
            raise HTTPException(status_code=400, detail="圖片錯誤")

        result = find_best_match_in_db(current_vector, products_ref)

        # -------------------------------------------------
        # ✅ success
        # -------------------------------------------------
        if result:
            doc_id, p_data, score = result

            return {
                "status": "success",
                "data": format_result(doc_id, p_data, score)
            }

        # -------------------------------------------------
        # ❌ not found（統一 schema）
        # -------------------------------------------------
        return {
            "status": "not_found",
            "data": {
                "id": "",
                "name": "無",
                "stock": 0,
                "image": "",
                "score": 0,
                "match": False,
                "location": "未設定"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 🔍 名稱搜尋（修正版）
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
            family_refs = db.collection("family_members").where(
                "userId", "==", user_id
            ).limit(1).get()

            if family_refs:
                target_id = family_refs[0].to_dict().get("familyId")
                id_field = "familyId"

        products_ref = db.collection("products") \
            .where(id_field, "==", target_id) \
            .stream()

        matches = []

        for doc in products_ref:
            p_data = doc.to_dict()
            name = p_data.get("name", "")

            if query_name.lower() in name.lower():
                matches.append({
                    "id": doc.id,
                    "name": name,
                    "stock": p_data.get("stock", 0),
                    "image": p_data.get("image", ""),
                    "score": 1.0,
                    "match": True,
                    "location": p_data.get("location", "未設定")
                })

        if matches:
            return {
                "status": "success",
                "data": matches
            }

        return {
            "status": "not_found",
            "data": []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "online"}
@app.get("/version")
def version():
    return {
        "version": "2026-05-02-main-v3",
        "status": "running",
        "note": "location fix + scan update"
    }

# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

print("Firebase project:", db.project)