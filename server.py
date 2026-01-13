from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime
import secrets

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'voicestudio')]

# Create the main app
app = FastAPI(title="Voice Studio Cologne API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBasic()

# Admin credentials (in production, use environment variables)
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'andrea')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'voicestudio2024')

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ============== MODELS ==============

class Workshop(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    date_start: str  # ISO date string (YYYY-MM-DD)
    date_end: Optional[str] = None
    time_start: Optional[str] = None  # Time string (HH:MM)
    time_end: Optional[str] = None  # Time string (HH:MM)
    location: str
    level: str  # Beginner, Intermediate, Advanced, All Levels
    max_participants: Optional[int] = None
    link: Optional[str] = None  # Optional external link/URL
    image_url: Optional[str] = None  # Optional image URL
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WorkshopCreate(BaseModel):
    title: str
    description: str
    date_start: str
    date_end: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    location: str
    level: str
    max_participants: Optional[int] = None
    link: Optional[str] = None
    image_url: Optional[str] = None

class NewsPost(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    image_url: Optional[str] = None  # Optional image URL
    is_published: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class NewsPostCreate(BaseModel):
    title: str
    content: str
    image_url: Optional[str] = None

class Registration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workshop_id: str
    name: str
    email: str
    phone: Optional[str] = None
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RegistrationCreate(BaseModel):
    workshop_id: str
    name: str
    email: str
    phone: Optional[str] = None
    message: Optional[str] = None

class Subscriber(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    subscribed_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class SubscriberCreate(BaseModel):
    email: str

class PushToken(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    token: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class PushTokenCreate(BaseModel):
    token: str

class SiteSettings(BaseModel):
    id: str = "site_settings"
    home_intro: str = "Welcome to Voice Studio Cologne"
    home_subtitle: str = "Discover your voice with Andrea Figallo"
    about_text: str = "Andrea Figallo is a vocal coach and choir director based in Cologne."
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SiteSettingsUpdate(BaseModel):
    home_intro: Optional[str] = None
    home_subtitle: Optional[str] = None
    about_text: Optional[str] = None

# ============== PUBLIC ROUTES ==============

@api_router.get("/")
async def root():
    return {"message": "Voice Studio Cologne API", "status": "running"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy"}

# Workshops
@api_router.get("/workshops", response_model=List[Workshop])
async def get_workshops():
    workshops = await db.workshops.find({"is_active": True}).sort("date_start", 1).to_list(100)
    return [Workshop(**w) for w in workshops]

@api_router.get("/workshops/{workshop_id}", response_model=Workshop)
async def get_workshop(workshop_id: str):
    workshop = await db.workshops.find_one({"id": workshop_id})
    if not workshop:
        raise HTTPException(status_code=404, detail="Workshop not found")
    return Workshop(**workshop)

# News
@api_router.get("/news", response_model=List[NewsPost])
async def get_news():
    posts = await db.news.find({"is_published": True}).sort("created_at", -1).to_list(50)
    return [NewsPost(**p) for p in posts]

@api_router.get("/news/{news_id}", response_model=NewsPost)
async def get_news_post(news_id: str):
    post = await db.news.find_one({"id": news_id})
    if not post:
        raise HTTPException(status_code=404, detail="News post not found")
    return NewsPost(**post)

# Registration
@api_router.post("/register", response_model=Registration)
async def create_registration(reg: RegistrationCreate):
    # Check if workshop exists
    workshop = await db.workshops.find_one({"id": reg.workshop_id, "is_active": True})
    if not workshop:
        raise HTTPException(status_code=404, detail="Workshop not found or inactive")
    
    # Check for duplicate registration
    existing = await db.registrations.find_one({
        "workshop_id": reg.workshop_id,
        "email": reg.email
    })
    if existing:
        raise HTTPException(status_code=400, detail="You are already registered for this workshop")
    
    registration = Registration(**reg.dict())
    await db.registrations.insert_one(registration.dict())
    return registration

# Mailing List
@api_router.post("/subscribe", response_model=Subscriber)
async def subscribe(sub: SubscriberCreate):
    # Check for existing subscriber
    existing = await db.subscribers.find_one({"email": sub.email})
    if existing:
        if existing.get("is_active"):
            raise HTTPException(status_code=400, detail="Email already subscribed")
        else:
            # Reactivate subscription
            await db.subscribers.update_one(
                {"email": sub.email},
                {"$set": {"is_active": True, "subscribed_at": datetime.utcnow()}}
            )
            return Subscriber(**{**existing, "is_active": True})
    
    subscriber = Subscriber(**sub.dict())
    await db.subscribers.insert_one(subscriber.dict())
    return subscriber

# Push Notifications Token
@api_router.post("/push-token", response_model=PushToken)
async def register_push_token(token_data: PushTokenCreate):
    existing = await db.push_tokens.find_one({"token": token_data.token})
    if existing:
        return PushToken(**existing)
    
    push_token = PushToken(**token_data.dict())
    await db.push_tokens.insert_one(push_token.dict())
    return push_token

# Site Settings (public read)
@api_router.get("/settings", response_model=SiteSettings)
async def get_settings():
    settings = await db.settings.find_one({"id": "site_settings"})
    if not settings:
        # Return defaults
        default_settings = SiteSettings()
        await db.settings.insert_one(default_settings.dict())
        return default_settings
    return SiteSettings(**settings)

# ============== ADMIN ROUTES ==============

@api_router.post("/admin/login")
async def admin_login(username: str = Depends(verify_admin)):
    return {"message": "Login successful", "username": username}

# Admin - Workshops
@api_router.get("/admin/workshops", response_model=List[Workshop])
async def admin_get_all_workshops(username: str = Depends(verify_admin)):
    workshops = await db.workshops.find().sort("date_start", 1).to_list(100)
    return [Workshop(**w) for w in workshops]

@api_router.post("/admin/workshops", response_model=Workshop)
async def admin_create_workshop(workshop: WorkshopCreate, username: str = Depends(verify_admin)):
    workshop_obj = Workshop(**workshop.dict())
    await db.workshops.insert_one(workshop_obj.dict())
    return workshop_obj

@api_router.put("/admin/workshops/{workshop_id}", response_model=Workshop)
async def admin_update_workshop(workshop_id: str, workshop: WorkshopCreate, username: str = Depends(verify_admin)):
    existing = await db.workshops.find_one({"id": workshop_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Workshop not found")
    
    update_data = workshop.dict()
    await db.workshops.update_one({"id": workshop_id}, {"$set": update_data})
    updated = await db.workshops.find_one({"id": workshop_id})
    return Workshop(**updated)

@api_router.delete("/admin/workshops/{workshop_id}")
async def admin_delete_workshop(workshop_id: str, username: str = Depends(verify_admin)):
    result = await db.workshops.delete_one({"id": workshop_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Workshop not found")
    return {"message": "Workshop deleted"}

@api_router.patch("/admin/workshops/{workshop_id}/toggle")
async def admin_toggle_workshop(workshop_id: str, username: str = Depends(verify_admin)):
    workshop = await db.workshops.find_one({"id": workshop_id})
    if not workshop:
        raise HTTPException(status_code=404, detail="Workshop not found")
    
    new_status = not workshop.get("is_active", True)
    await db.workshops.update_one({"id": workshop_id}, {"$set": {"is_active": new_status}})
    return {"message": f"Workshop {'activated' if new_status else 'deactivated'}"}

# Admin - News
@api_router.get("/admin/news", response_model=List[NewsPost])
async def admin_get_all_news(username: str = Depends(verify_admin)):
    posts = await db.news.find().sort("created_at", -1).to_list(100)
    return [NewsPost(**p) for p in posts]

@api_router.post("/admin/news", response_model=NewsPost)
async def admin_create_news(post: NewsPostCreate, username: str = Depends(verify_admin)):
    news_obj = NewsPost(**post.dict())
    await db.news.insert_one(news_obj.dict())
    return news_obj

@api_router.put("/admin/news/{news_id}", response_model=NewsPost)
async def admin_update_news(news_id: str, post: NewsPostCreate, username: str = Depends(verify_admin)):
    existing = await db.news.find_one({"id": news_id})
    if not existing:
        raise HTTPException(status_code=404, detail="News post not found")
    
    update_data = post.dict()
    await db.news.update_one({"id": news_id}, {"$set": update_data})
    updated = await db.news.find_one({"id": news_id})
    return NewsPost(**updated)

@api_router.delete("/admin/news/{news_id}")
async def admin_delete_news(news_id: str, username: str = Depends(verify_admin)):
    result = await db.news.delete_one({"id": news_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="News post not found")
    return {"message": "News post deleted"}

@api_router.patch("/admin/news/{news_id}/toggle")
async def admin_toggle_news(news_id: str, username: str = Depends(verify_admin)):
    post = await db.news.find_one({"id": news_id})
    if not post:
        raise HTTPException(status_code=404, detail="News post not found")
    
    new_status = not post.get("is_published", True)
    await db.news.update_one({"id": news_id}, {"$set": {"is_published": new_status}})
    return {"message": f"News post {'published' if new_status else 'unpublished'}"}

# Admin - Registrations
@api_router.get("/admin/registrations", response_model=List[Registration])
async def admin_get_registrations(username: str = Depends(verify_admin)):
    registrations = await db.registrations.find().sort("created_at", -1).to_list(500)
    return [Registration(**r) for r in registrations]

@api_router.get("/admin/registrations/{workshop_id}", response_model=List[Registration])
async def admin_get_workshop_registrations(workshop_id: str, username: str = Depends(verify_admin)):
    registrations = await db.registrations.find({"workshop_id": workshop_id}).sort("created_at", -1).to_list(500)
    return [Registration(**r) for r in registrations]

@api_router.delete("/admin/registrations/{registration_id}")
async def admin_delete_registration(registration_id: str, username: str = Depends(verify_admin)):
    result = await db.registrations.delete_one({"id": registration_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Registration not found")
    return {"message": "Registration deleted"}

# Admin - Subscribers
@api_router.get("/admin/subscribers", response_model=List[Subscriber])
async def admin_get_subscribers(username: str = Depends(verify_admin)):
    subscribers = await db.subscribers.find().sort("subscribed_at", -1).to_list(1000)
    return [Subscriber(**s) for s in subscribers]

@api_router.delete("/admin/subscribers/{subscriber_id}")
async def admin_delete_subscriber(subscriber_id: str, username: str = Depends(verify_admin)):
    result = await db.subscribers.delete_one({"id": subscriber_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    return {"message": "Subscriber deleted"}

# Admin - Export subscribers as CSV
@api_router.get("/admin/subscribers/export")
async def admin_export_subscribers(username: str = Depends(verify_admin)):
    subscribers = await db.subscribers.find({"is_active": True}).to_list(10000)
    csv_content = "email,subscribed_at\n"
    for s in subscribers:
        csv_content += f"{s['email']},{s['subscribed_at']}\n"
    return {"csv": csv_content, "count": len(subscribers)}

# Admin - Site Settings
@api_router.put("/admin/settings", response_model=SiteSettings)
async def admin_update_settings(settings: SiteSettingsUpdate, username: str = Depends(verify_admin)):
    update_data = {k: v for k, v in settings.dict().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()
    
    await db.settings.update_one(
        {"id": "site_settings"},
        {"$set": update_data},
        upsert=True
    )
    
    updated = await db.settings.find_one({"id": "site_settings"})
    return SiteSettings(**updated)

# Admin - Dashboard Stats
@api_router.get("/admin/stats")
async def admin_get_stats(username: str = Depends(verify_admin)):
    workshops_count = await db.workshops.count_documents({"is_active": True})
    registrations_count = await db.registrations.count_documents({})
    subscribers_count = await db.subscribers.count_documents({"is_active": True})
    news_count = await db.news.count_documents({"is_published": True})
    
    return {
        "active_workshops": workshops_count,
        "total_registrations": registrations_count,
        "active_subscribers": subscribers_count,
        "published_news": news_count
    }

# Admin - Send Push Notification
class PushNotificationRequest(BaseModel):
    title: str
    body: str
    workshop_id: Optional[str] = None

@api_router.post("/admin/send-notification")
async def admin_send_notification(notification: PushNotificationRequest, username: str = Depends(verify_admin)):
    import httpx
    
    # Get all active push tokens
    tokens = await db.push_tokens.find({"is_active": True}).to_list(1000)
    
    if not tokens:
        return {"message": "No push tokens registered", "sent": 0}
    
    # Expo Push API endpoint
    expo_push_url = "https://exp.host/--/api/v2/push/send"
    
    messages = []
    for token_doc in tokens:
        token = token_doc.get("token")
        if token and token.startswith("ExponentPushToken"):
            message = {
                "to": token,
                "sound": "default",
                "title": notification.title,
                "body": notification.body,
            }
            if notification.workshop_id:
                message["data"] = {"workshopId": notification.workshop_id}
            messages.append(message)
    
    if not messages:
        return {"message": "No valid Expo push tokens found", "sent": 0}
    
    # Send to Expo Push API
    sent_count = 0
    failed_tokens = []
    
    async with httpx.AsyncClient() as client_http:
        # Send in batches of 100 (Expo limit)
        for i in range(0, len(messages), 100):
            batch = messages[i:i+100]
            try:
                response = await client_http.post(
                    expo_push_url,
                    json=batch,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    result = response.json()
                    data = result.get("data", [])
                    for j, ticket in enumerate(data):
                        if ticket.get("status") == "ok":
                            sent_count += 1
                        elif ticket.get("status") == "error":
                            # Mark invalid tokens as inactive
                            if ticket.get("details", {}).get("error") in ["DeviceNotRegistered", "InvalidCredentials"]:
                                failed_tokens.append(batch[j]["to"])
            except Exception as e:
                logger.error(f"Error sending push notification batch: {e}")
    
    # Deactivate failed tokens
    if failed_tokens:
        await db.push_tokens.update_many(
            {"token": {"$in": failed_tokens}},
            {"$set": {"is_active": False}}
        )
    
    return {
        "message": f"Notification sent to {sent_count} devices",
        "sent": sent_count,
        "total_tokens": len(messages),
        "failed": len(failed_tokens)
    }

# Serve ZIP file for download
from fastapi.responses import FileResponse

@api_router.get("/download-project")
async def download_project():
    file_path = "/app/backend/voice-studio-cologne.zip"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename="voice-studio-cologne.zip",
            media_type="application/zip"
        )
    raise HTTPException(status_code=404, detail="File not found")

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
