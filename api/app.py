from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import yaml
import json
from datetime import datetime
import asyncio
from web3 import Web3
import os
from dotenv import load_dotenv

from .routes import models, verify
from blockchain.interact import BlockchainInteractor

load_dotenv()

app = FastAPI(
    title="Blockchain-Verifiable ML Models API",
    description="API for registering and verifying ML models on blockchain",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize blockchain interactor
blockchain_interactor = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global blockchain_interactor
    
    try:
        blockchain_interactor = BlockchainInteractor()
        print("✅ Blockchain interactor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize blockchain: {e}")
        blockchain_interactor = None

# Include routers
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(verify.router, prefix="/api/v1/verify", tags=["verification"])

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    blockchain_connected: bool
    version: str

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    blockchain_status = blockchain_interactor is not None and blockchain_interactor.w3.is_connected()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        blockchain_connected=blockchain_status,
        version="1.0.0"
    )

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    checks = {
        "api": True,
        "database": check_database(),
        "blockchain": blockchain_interactor.w3.is_connected() if blockchain_interactor else False,
        "storage": check_storage(),
        "timestamp": datetime.now().isoformat()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks
    }

class ModelRegistrationRequest(BaseModel):
    model_hash: str
    metadata_hash: str
    version: str
    ipfs_cid: Optional[str] = None
    model_path: Optional[str] = None

@app.post("/api/v1/register")
async def register_model(request: ModelRegistrationRequest, background_tasks: BackgroundTasks):
    """Register a model on blockchain"""
    if not blockchain_interactor:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Check if model already registered
        if blockchain_interactor.is_model_registered(request.model_hash):
            raise HTTPException(status_code=400, detail="Model already registered")
        
        # Register on blockchain
        tx_hash = blockchain_interactor.register_model(
            model_hash=request.model_hash,
            metadata_hash=request.metadata_hash,
            version=request.version,
            ipfs_cid=request.ipfs_cid
        )
        
        # Background task to verify transaction
        background_tasks.add_task(
            wait_for_confirmation,
            tx_hash,
            request.model_hash
        )
        
        return {
            "status": "pending",
            "transaction_hash": tx_hash,
            "message": "Model registration submitted. Verification in progress."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/model/{model_hash}")
async def get_model_info(model_hash: str):
    """Get model information from blockchain"""
    if not blockchain_interactor:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        record = blockchain_interactor.get_model_record(model_hash)
        
        if not record:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "model_hash": model_hash,
            "metadata_hash": record[1],
            "owner": record[2],
            "timestamp": record[3],
            "block_number": record[4],
            "version": record[5],
            "ipfs_cid": record[6],
            "verified": record[7]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/owner/{owner_address}")
async def get_owner_models(owner_address: str):
    """Get all models owned by an address"""
    if not blockchain_interactor:
        raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    
    try:
        # Validate address
        if not Web3.is_address(owner_address):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")
        
        models = blockchain_interactor.get_models_by_owner(owner_address)
        
        return {
            "owner": owner_address,
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def wait_for_confirmation(tx_hash: str, model_hash: str):
    """Background task to wait for transaction confirmation"""
    await asyncio.sleep(30)  # Wait for blocks
    
    if blockchain_interactor:
        try:
            receipt = blockchain_interactor.w3.eth.wait_for_transaction_receipt(
                tx_hash, 
                timeout=120
            )
            
            if receipt.status == 1:
                print(f"✅ Transaction confirmed for model {model_hash}")
            else:
                print(f"❌ Transaction failed for model {model_hash}")
                
        except Exception as e:
            print(f"Error confirming transaction: {e}")

def check_database():
    """Check database connectivity"""
    try:
        # Implement actual database check
        return True
    except:
        return False

def check_storage():
    """Check storage availability"""
    try:
        storage_path = config['model']['storage_path']
        return os.path.exists(storage_path) and os.access(storage_path, os.W_OK)
    except:
        return False

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug'],
        workers=config['api']['workers']
    )