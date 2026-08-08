from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import DATABASE_URL
from db.models import Base


engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # log SQL statements to the console when True
    future=True,  # use SQLAlchemy 2.0 style behavior
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,  # seconds to wait for a connection from the pool
    pool_recycle=1800,  # seconds to recycle connections and avoid stale connections
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    """Create SQLAlchemy tables in PostgreSQL when the app starts."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async dependency that yields a session from the connection pool."""
    async with AsyncSessionLocal() as session:
        yield session
