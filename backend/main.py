from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship

DATABASE_URL = os.getenv("DB_URL", "postgresql+psycopg2://vodnet:vodnet@localhost:5432/vodnet")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ext_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    map: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    events = relationship("Event", back_populates="analysis", cascade="all, delete-orphan")


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detection_method: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    analysis = relationship("Analysis", back_populates="events")


# Pydantic schemas
class EventIn(BaseModel):
    timestamp_ms: int
    event_type: str
    confidence: Optional[float] = None
    detection_method: Optional[str] = None
    details: Optional[dict] = None


class AnalysisCreate(BaseModel):
    ext_id: str = Field(..., description="External identifier, e.g. video folder name")
    map: Optional[str] = None
    created_ms: Optional[int] = None
    events: List[EventIn] = Field(default_factory=list)


class AnalysisOut(BaseModel):
    id: int
    ext_id: str
    map: Optional[str]
    created_ms: Optional[int]
    count: int


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Create tables
    Base.metadata.create_all(bind=engine)


app = FastAPI(title="VOD-Net Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/analyses", response_model=AnalysisOut)
def create_analysis(payload: AnalysisCreate, db=Depends(get_db)):
    existing = db.query(Analysis).filter(Analysis.ext_id == payload.ext_id).one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Analysis already exists")
    a = Analysis(ext_id=payload.ext_id, map=payload.map, created_ms=payload.created_ms)
    db.add(a)
    db.flush()
    if payload.events:
        batch = [
            Event(
                analysis_id=a.id,
                timestamp_ms=e.timestamp_ms,
                event_type=e.event_type,
                confidence=e.confidence,
                detection_method=e.detection_method,
                details=e.details,
            )
            for e in payload.events
        ]
        db.add_all(batch)
    db.commit()
    count = db.query(Event).filter(Event.analysis_id == a.id).count()
    return AnalysisOut(id=a.id, ext_id=a.ext_id, map=a.map, created_ms=a.created_ms, count=count)


@app.get("/analyses", response_model=List[AnalysisOut])
def list_analyses(limit: int = 50, db=Depends(get_db)):
    rows = (
        db.query(Analysis.id, Analysis.ext_id, Analysis.map, Analysis.created_ms)
        .order_by(Analysis.id.desc())
        .limit(limit)
        .all()
    )
    out: List[AnalysisOut] = []
    for r in rows:
        count = db.query(Event).filter(Event.analysis_id == r.id).count()
        out.append(AnalysisOut(id=r.id, ext_id=r.ext_id, map=r.map, created_ms=r.created_ms, count=count))
    return out


@app.get("/analyses/{ext_id}")
def get_analysis(ext_id: str, db=Depends(get_db)):
    a = db.query(Analysis).filter(Analysis.ext_id == ext_id).one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Not found")
    events = (
        db.query(Event)
        .filter(Event.analysis_id == a.id)
        .order_by(Event.timestamp_ms.asc())
        .all()
    )
    return {
        "id": a.id,
        "ext_id": a.ext_id,
        "map": a.map,
        "created_ms": a.created_ms,
        "events": [
            {
                "timestamp_ms": e.timestamp_ms,
                "event_type": e.event_type,
                "confidence": e.confidence,
                "detection_method": e.detection_method,
                "details": e.details,
            }
            for e in events
        ],
    }


class EventsAppend(BaseModel):
    events: List[EventIn]


@app.post("/analyses/{ext_id}/events")
def append_events(ext_id: str, payload: EventsAppend, db=Depends(get_db)):
    a = db.query(Analysis).filter(Analysis.ext_id == ext_id).one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Not found")
    if payload.events:
        batch = [
            Event(
                analysis_id=a.id,
                timestamp_ms=e.timestamp_ms,
                event_type=e.event_type,
                confidence=e.confidence,
                detection_method=e.detection_method,
                details=e.details,
            )
            for e in payload.events
        ]
        db.add_all(batch)
        db.commit()
    return {"ok": True}

